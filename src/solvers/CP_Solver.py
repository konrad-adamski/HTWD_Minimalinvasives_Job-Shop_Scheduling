import contextlib
import os
import sys
from fractions import Fraction
from typing import Optional
from ortools.sat.python import cp_model

from src.Logger import Logger
from src.domain.Collection import LiveJobCollection
from src.domain.orm_models import JobOperation
from src.solvers.CP_BoundStagnationGuard import BoundGuard
from src.solvers.CP_Collections import MachineFixIntervalMap, OperationIndexMapper, JobDelayMap, MachineFixInterval, \
    StartTimes, EndTimes, Intervals, OriginalOperationStarts, CostVarCollection


class Solver:

    def __init__(self, jobs_collection: LiveJobCollection, logger: Logger, schedule_start: int = 0):

        self.logger = logger

        # JobsCollections and information
        self.jobs_collection = jobs_collection
        self.previous_schedule_jobs_collection = None
        self.active_jobs_collection = None

        self.machines = jobs_collection.get_unique_machine_names()
        self.schedule_start = schedule_start

        # Model and solver
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()

        self.solver_status = None
        self.model_completed: bool = False

        # Cost collections
        self.tardiness_terms = CostVarCollection()
        self.earliness_terms = CostVarCollection()
        self.deviation_terms = CostVarCollection()

        #  Variable collections
        self.index_mapper = OperationIndexMapper()
        self.start_times = StartTimes()
        self.end_times = EndTimes()
        self.intervals = Intervals()

        # for active operations
        self.machines_fix_intervals = MachineFixIntervalMap()
        self.job_delays = JobDelayMap()

        # for previous schedule operations starts
        self.original_operation_starts = OriginalOperationStarts()

        # Horizon (Worst-case upper bound)--------------------------------------------------------------
        total_duration = jobs_collection.get_total_duration()

        if jobs_collection.get_latest_due_date():
            known_highest_value = jobs_collection.get_latest_due_date()
        else:
            known_highest_value = jobs_collection.get_latest_earliest_start()
        self.horizon = known_highest_value + total_duration

        # Create Variables -----------------------------------------------------------------------------
        jobs_collection.sort_operations()
        jobs_collection.sort_jobs_by_arrival()

        for job_idx, job in enumerate(jobs_collection.values()):
            for op_idx, operation in enumerate(job.operations):
                suffix = f"{job_idx}_{op_idx}"
                start = self.model.NewIntVar(job.earliest_start, self.horizon, f"start_{suffix}")
                end = self.model.NewIntVar(job.earliest_start, self.horizon, f"end_{suffix}")

                interval = self.model.NewIntervalVar(start, operation.duration, end, f"interval_{suffix}")
                # interval = model.NewIntervalVar(start, operation.duration, start + operation.duration, f"interval_{suffix}")

                self.start_times[(job_idx, op_idx)] = start
                self.end_times[(job_idx, op_idx)] = end
                self.intervals[(job_idx, op_idx)] = (interval, operation.machine_name)
                self.index_mapper.add(job_idx, op_idx, operation)


    # Rescheduling -------------------------------------------------------------------------------------------
    def _extract_previous_starts_for_deviation(self):
        # Previous schedule: extract start times for deviation penalties
        if self.previous_schedule_jobs_collection is not None:
            for job in self.previous_schedule_jobs_collection.values():
                for operation in job.operations:
                    index = self.index_mapper.get_index_from_operation(operation)
                    if index is not None:
                        job_idx, op_idx = index
                        self.original_operation_starts[(job_idx, op_idx)] = operation.start

    def _extract_delays_from_active_operations(self):
        # Active operations: block machines and delay jobs
        if self.active_jobs_collection is not None:
            for job in self.active_jobs_collection.values():
                for operation in job.operations:
                    self.machines_fix_intervals.update_interval(
                        machine=operation.machine_name,
                        start=self.schedule_start,
                        end=operation.end
                    )
                    self.job_delays.update_delay(job_id=job.id, time_stamp=operation.end)

    # Constraints ---------------------------------------------------------------------------------------------------

    def _add_technological_operation_constraints(self):

        for (job_idx, op_idx), operation in self.index_mapper.items():
            start_var = self.start_times[(job_idx, op_idx)]

            # 1. Technological constraint: earliest start of the first operation
            if op_idx == 0:
                min_start = max(operation.job_earliest_start, int(self.schedule_start))
                if operation.job_id in self.job_delays:
                    min_start = max(min_start, self.job_delays.get_time(operation.job_id))
                self.model.Add(start_var >= min_start)

            # 2. Technological constraint: operation order within the job
            if op_idx > 0:
                self.model.Add(start_var >= self.end_times[(job_idx, op_idx - 1)])


    def _add_technological_operation_constraints_with_transition_times(self):

        for (job_idx, op_idx), operation in self.index_mapper.items():
            start_var = self.start_times[(job_idx, op_idx)]

            # 1. Technological constraint: earliest start of the first operation
            if op_idx == 0:
                min_start = max(operation.job_earliest_start, int(self.schedule_start))

                if operation.position_number ==  0:                                            #  oder Datenbankabfrage!
                    due_date = operation.job_due_date
                    left_transition_time = operation.job.sum_left_transition_time(operation.position_number)
                    duration = operation.job.sum_duration
                    reasonable_min_start = due_date - duration - left_transition_time
                    min_start = max(min_start, reasonable_min_start)

                if operation.job_id in self.job_delays:
                    min_start = max(min_start, self.job_delays.get_time(operation.job_id))
                self.model.Add(start_var >= min_start)

            # 2. Technological constraint: operation order within the job
            if op_idx > 0:
                self.model.Add(start_var >= self.end_times[(job_idx, op_idx - 1)])


    def _add_machine_no_overlap_constraints(self):
        """
        If rescheduling: first add machines_fix_intervals to the solver
        - self.machines
        - self.intervals
        - self.machines_fix_intervals (from active_jobs_collection) - optional
        """

        # Machine-level constraints (no overlap + fixed blocks from running ops) -----------------------
        for machine in self.machines:
            machine_intervals = []

            # Intervals of the operations that are planned on this machine
            for (_, _), (interval, machine_name) in self.intervals.items():
                if machine_name == machine:
                    machine_intervals.append(interval)

            # Fixed Intervals of active operations from previous shift
            if self.active_jobs_collection and machine in self.machines_fix_intervals:
                machine_fix_interval = self.machines_fix_intervals[machine]
                start = machine_fix_interval.start
                end = machine_fix_interval.end
                if start < end:
                    fixed_interval = self.model.NewIntervalVar(start, end - start, end, f"fixed_{machine}")
                    machine_intervals.append(fixed_interval)

            # NoOverlap für diese Maschine
            self.model.AddNoOverlap(machine_intervals)


    def _add_tardiness_var(self, job_idx: int, op_idx: int, operation: JobOperation):
        if operation.position_number != operation.job.last_operation_position_number:
            raise ValueError(f"{operation} is not the last operation! '_add_tardiness_var()' failed!")
        end_var = self.end_times[(job_idx, op_idx)]
        tardiness = self.model.NewIntVar(0, self.horizon, f"tardiness_{job_idx}")
        self.model.AddMaxEquality(tardiness, [end_var - operation.job_due_date, 0])

        self.tardiness_terms.add(tardiness)

    def _add_earliness_var(self, job_idx: int, op_idx: int, operation: JobOperation):
        if operation.position_number != operation.job.last_operation_position_number:
            raise ValueError(f"{operation} is not the last operation! '_add_earliness_var()' failed!")

        end_var = self.end_times[(job_idx, op_idx)]
        earliness = self.model.NewIntVar(0, self.horizon, f"earliness_{job_idx}")
        self.model.AddMaxEquality(earliness, [operation.job_due_date - end_var, 0])
        self.earliness_terms.add(earliness)

    def _add_start_deviation_var(self, job_idx: int, op_idx: int):
        start_var = self.start_times[(job_idx, op_idx)]

        if (job_idx, op_idx) in self.original_operation_starts.keys():
            deviation = self.model.NewIntVar(0, self.horizon, f"deviation_{job_idx}_{op_idx}")
            original_start = self.original_operation_starts[(job_idx, op_idx)]
            self.model.AddAbsEquality(deviation, start_var - original_start)
            self.deviation_terms.add(deviation)


    # Main model builder -----------------------------------------------------------------------------------------------
    def build_model__absolute_lateness__start_deviation__minimization(
            self, previous_schedule_jobs_collection: Optional[LiveJobCollection] = None,
            active_jobs_collection: Optional[LiveJobCollection] = None,
            w_t: int = 1, w_e: int = 1, w_dev: int = 1):
        # with_transition_times for first operation

        if self.model_completed:
            self.logger.warning("Model already completed!")
            return False

        self.logger.info("Building model for absolute lateness and start deviation minimization")
        self.previous_schedule_jobs_collection = previous_schedule_jobs_collection
        self.active_jobs_collection = active_jobs_collection

        # I. Extractions from previous schedule and simulation!
        self._extract_previous_starts_for_deviation()
        self._extract_delays_from_active_operations()

        # II. Constraints (after I.)
        self._add_machine_no_overlap_constraints()
        self._add_technological_operation_constraints_with_transition_times()

        # III. Operation-level variables (after I.)
        for (job_idx, op_idx), operation in self.index_mapper.items():

            # Lateness terms for the job (last operation)
            if operation.position_number == operation.job.last_operation_position_number:
                # Tardiness
                self._add_tardiness_var(job_idx, op_idx, operation)

                # Earliness
                self._add_earliness_var(job_idx, op_idx, operation)

            # Deviation from original schedule
            self._add_start_deviation_var(job_idx, op_idx)

        # IV. Weights
        if previous_schedule_jobs_collection is None or previous_schedule_jobs_collection.count_operations() == 0:
            w_dev = 0
        self.logger.info(f"Model weights: {w_t = }, {w_e = }, {w_dev = }")

        self.tardiness_terms.set_weight(weight=w_t)
        self.earliness_terms.set_weight(weight=w_e)
        self.deviation_terms.set_weight(weight=w_dev)

        # V. Objective function
        self.model.Minimize(
            self.tardiness_terms.objective_expr()
            + self.earliness_terms.objective_expr()
            + self.deviation_terms.objective_expr()
        )
        self.model_completed = True


    # Makespan ---------------------------------------------------------------------------------------------------------
    def build_makespan_model(self):
        if self.model_completed:
            return "Model is already completed"

        # Operation-level constraints
        self._add_technological_operation_constraints()

        #  Machine-level constraints
        self._add_machine_no_overlap_constraints()

        makespan = self.model.NewIntVar(0, self.horizon, "makespan")
        for (job_idx, op_idx), operation in self.index_mapper.items():
            if operation.position_number == operation.job.last_operation_position_number:
                self.model.Add(makespan >= self.end_times[(job_idx, op_idx)])
        self.model.Minimize(makespan)

        self.model_completed = True

    # Legacy absolute Lateness with w_first (any without transition times)
    def build_model__absolute_lateness__first_operation_earliness__start_deviation__minimization(
            self, previous_schedule_jobs_collection: Optional[LiveJobCollection] = None,
            active_jobs_collection: Optional[LiveJobCollection] = None,
            w_t: int = 1, w_e: int = 1, w_first: int = 1, w_dev: int = 1,
            duration_buffer_factor: float = 2.0):

        if self.model_completed:
            return "Model is already completed"

        self.previous_schedule_jobs_collection = previous_schedule_jobs_collection
        self.active_jobs_collection = active_jobs_collection

        # First operation earliness variables
        first_op_terms = CostVarCollection()

        # I. Extractions from previous schedule and simulation!
        self._extract_previous_starts_for_deviation()
        self._extract_delays_from_active_operations()

        # II. Constraints (after I.)
        self._add_machine_no_overlap_constraints()
        self._add_technological_operation_constraints()

        # III. Operation-level variables (after I.)
        for (job_idx, op_idx), operation in self.index_mapper.items():
            start_var = self.start_times[(job_idx, op_idx)]
            if operation.position_number ==  0:                                                #  oder Datenbankabfrage!

                first_op_latest_desired_start = int(
                    operation.job_due_date - operation.job.sum_duration * duration_buffer_factor)
                first_op_latest_desired_start = max(self.schedule_start, first_op_latest_desired_start)
                first_op_earliness = self.model.NewIntVar(0, self.horizon, f"first_op_earliness_{job_idx}")
                self.model.AddMaxEquality(first_op_earliness, [first_op_latest_desired_start - start_var, 0])

                first_op_terms.add(first_op_earliness)

            # Lateness terms for the job (last operation)
            if operation.position_number == operation.job.last_operation_position_number:

                # Tardiness
                self._add_tardiness_var(job_idx, op_idx, operation)

                # Earliness
                self._add_earliness_var(job_idx, op_idx, operation)

            # Deviation from original schedule
            self._add_start_deviation_var(job_idx, op_idx)

        # IV. Weights
        if previous_schedule_jobs_collection is None or previous_schedule_jobs_collection.count_operations() == 0:
            w_dev = 0

        self.logger.info(f"Model weights: {w_t = }, {w_e = }, {w_first = }, {w_dev = }")

        self.tardiness_terms.set_weight(weight=w_t)
        self.earliness_terms.set_weight(weight=w_e)
        first_op_terms.set_weight(weight=w_first)
        self.deviation_terms.set_weight(weight=w_dev)

        # V. Objective function
        self.model.Minimize(
            self.tardiness_terms.objective_expr()
            + self.earliness_terms.objective_expr()
            + first_op_terms.objective_expr()
            + self.deviation_terms.objective_expr()
        )

        self.model_completed = True

    # Legacy flowtime
    def build_model__flowtime__start_deviation__minimization(
            self, previous_schedule_jobs_collection: Optional[LiveJobCollection] = None,
            active_jobs_collection: Optional[LiveJobCollection] = None, w_f: int = 1, w_dev: int = 1):

        if self.model_completed:
            return "Model is already completed"

        self.previous_schedule_jobs_collection = previous_schedule_jobs_collection
        self.active_jobs_collection = active_jobs_collection

        # Flowtime variables
        flowtime_terms = CostVarCollection()

        # I. Extractions from previous schedule and simulation!
        self._extract_previous_starts_for_deviation()
        self._extract_delays_from_active_operations()

        # II. Constraints (after I.)
        self._add_machine_no_overlap_constraints()
        self._add_technological_operation_constraints()

        # III. Operation-level variables (after I.)
        for (job_idx, op_idx), operation in self.index_mapper.items():
            end_var = self.end_times[(job_idx, op_idx)]

            # FlowTime (only last operation)
            if operation.position_number == operation.job.last_operation_position_number:
                flowtime = self.model.NewIntVar(0, self.horizon, f"flowtime_{job_idx}")
                self.model.Add(flowtime == end_var - operation.job_earliest_start)
                flowtime_terms.add(flowtime)

            # Deviation from original schedule
            self._add_start_deviation_var(job_idx, op_idx)

        # IV. Weights
        if previous_schedule_jobs_collection is None or previous_schedule_jobs_collection.count_operations() == 0:
            w_dev = 0

        self.logger.info(f"Model weights: {w_f = }, {w_dev = }")

        flowtime_terms.set_weight(weight=w_f)
        self.deviation_terms.set_weight(weight=w_dev)

        # V. Objective function
        self.model.Minimize(flowtime_terms.objective_expr() + self.deviation_terms.objective_expr())
        self.model_completed = True

    def solve_model(
            self,
            print_log_search_progress: bool = False,
            time_limit: Optional[int] = None,
            gap_limit: float = 0.0,
            log_file: Optional[str] = None,
            bound_no_improvement_time: Optional[int] = 600,
            bound_relative_change: float = 0.01,
            bound_warmup_time: int = 30,
    ):
        if self.model_completed:

            self.solver.parameters.num_search_workers = int(os.environ.get("MAX_CPU_NUMB", "8"))

            self.solver.parameters.log_search_progress = print_log_search_progress
            self.solver.parameters.relative_gap_limit = gap_limit

            if time_limit is not None:
                self.solver.parameters.max_time_in_seconds = time_limit

            # Bound-Callback vorbereiten
            if bound_no_improvement_time is not None and bound_no_improvement_time > 0:
                self.solver.best_bound_callback = BoundGuard(
                    solver=self.solver,
                    logger=self.logger,
                    no_improvement_seconds=bound_no_improvement_time,
                    warmup_seconds=bound_warmup_time,
                    relative_change=bound_relative_change,
                )
            if log_file is not None:
                # Für Log-Ausgabe ins File aktivieren
                self.solver.parameters.log_search_progress = True
                with _redirect_cpp_logs(log_file):
                    self.solver_status = self.solver.Solve(self.model)
            else:
                self.solver_status = self.solver.Solve(self.model)

        else:
            self.logger.warning("Model was not completed yet.")

    def get_schedule(self):

        if self.solver_status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            schedule_job_collection = LiveJobCollection()

            for (job_idx, op_idx), operation in self.index_mapper.items():
                start = self.solver.Value(self.start_times[(job_idx, op_idx)])
                end = self.solver.Value(self.end_times[(job_idx, op_idx)])

                schedule_job_collection.add_operation_instance(
                    op=operation,
                    new_start=start,
                    new_end=end
                )

            return schedule_job_collection


    def get_model_info(self):
        if self.model_completed:
            model_proto = self.model.Proto()
            model_info = {
                "number_of_preparable_operations": self.jobs_collection.count_operations(),
                "number_of_previous_operations": self.previous_schedule_jobs_collection.count_operations() if self.previous_schedule_jobs_collection else 0,
                "number_of_active_operation": self.active_jobs_collection.count_operations() if self.active_jobs_collection else 0,
                "number_of_variables": len(model_proto.variables),
                "number_of_constraints": len(model_proto.constraints)
            }
            return model_info
        return {"access_fault": "Model is not complete!"}

    def get_solver_info(self) -> dict:
        if self.solver_status:
            solver_info = {
                "status": self.solver.StatusName(self.solver_status),
                "objective_value": self.solver.ObjectiveValue() if self.solver_status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None,
                "best_objective_bound": self.solver.BestObjectiveBound(),
                "number_of_branches": self.solver.NumBranches(),
                "wall_time": round(self.solver.WallTime(), 2)
            }
            if self.solver_status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                solver_info["tardiness_cost"] = self.tardiness_terms.total_cost(self.solver)
                solver_info["earliness_cost"] = self.earliness_terms.total_cost(self.solver)
                solver_info["deviation_cost"] = self.deviation_terms.total_cost(self.solver)

            return solver_info
        return {"access_fault": "Solver status is not available!"}


    def log_model_info(self):
        self.logger.info("Model info "+ "-"*15)
        self._log_info(self.get_model_info(), label_width= 31)

    def log_solver_info(self):
        self.logger.info("Solver info "+ "-"*14)
        self._log_info(self.get_solver_info())

    def _log_info(self, info: dict, label_width: int = 20):
        """
        Pretty log a dictionary.
        Replaces underscores with spaces and aligns keys.
        """
        for key, value in info.items():
            label = key.replace("_", " ").capitalize()
            self.logger.info(f"{label:{label_width}}: {value}")


@contextlib.contextmanager
def _redirect_cpp_logs(logfile_path: str = "cp_output.log"):
    """
    Context manager to temporarily redirect stdout/stderr,
    e.g. to capture output from OR-Tools CP-SAT solver or other C++ logs.
    After the block, original output streams are restored.
    """

    # Flush any current output to avoid mixing content
    sys.stdout.flush()
    sys.stderr.flush()

    # Save original file descriptors for stdout and stderr
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)

    with open(logfile_path, 'w') as f:
        try:
            # Redirect stdout and stderr to the log file
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
            yield
            f.flush()  # Ensures content is flushed to file, esp. in Jupyter
        finally:
            # Restore original stdout and stderr
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)


