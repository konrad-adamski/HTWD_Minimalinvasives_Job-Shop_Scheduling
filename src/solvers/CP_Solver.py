import contextlib
import os
import sys
from fractions import Fraction
from typing import Optional

from ortools.sat.cp_model_pb2 import CpSolverStatus
from ortools.sat.python import cp_model

from src.domain.Collection import LiveJobCollection
from src.solvers.CP_Collections import MachineFixIntervalMap, OperationIndexMapper, JobDelayMap, MachineFixInterval, \
    StartTimes, EndTimes, Intervals, OriginalOperationStarts


class Solver:

    def __init__(self, jobs_collection: LiveJobCollection, schedule_start: int = 0):

        self.jobs_collection = jobs_collection
        self.previous_schedule_jobs_collection = None
        self.active_jobs_collection = None
        self.solver_status = None
        self.model_completed: bool = False
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()

        self.index_mapper = OperationIndexMapper()
        self.start_times = StartTimes()
        self.end_times = EndTimes()
        self.intervals = Intervals()

        # Objects for active operations
        self.machines_fix_intervals = MachineFixIntervalMap()
        self.job_delays = JobDelayMap()

        # Objects for previous schedule operations starts
        self.original_operation_starts = OriginalOperationStarts()

        self.schedule_start = schedule_start
        self.machines = jobs_collection.get_unique_machine_names()

        # Horizon (Worst-case upper bound)--------------------------------------------------------------
        total_duration = jobs_collection.get_total_duration()

        if jobs_collection.get_latest_deadline():
            known_highest_value = jobs_collection.get_latest_deadline()
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


    def build_makespan_model(self):
        if self.model_completed:
            return "Model is already completed"

        # Operation-level constraints ------------------------------------------------------------------
        self._add_technological_operation_constraints()

        #  Machine-level constraints -------------------------------------------------------------------
        for machine in self.machines:
            machine_intervals = []

            # Add operation intervals for this machine
            for (_, _), (interval, machine_name) in self.intervals.items():
                if machine_name == machine:
                    machine_intervals.append(interval)

            # NoOverlap for this machine
            self.model.AddNoOverlap(machine_intervals)

        makespan = self.model.NewIntVar(0, self.horizon, "makespan")
        for (job_idx, op_idx), operation in self.index_mapper.items():
            if operation.position_number == operation.job.last_operation_position_number:
                self.model.Add(makespan >= self.end_times[(job_idx, op_idx)])
        self.model.Minimize(makespan)

        self.model_completed = True


    # With previous schedule information and delay information (due to active operations)
    def _build_reschedule_model(self):

        # Previous schedule: extract start times for deviation penalties -------------------------------
        if self.previous_schedule_jobs_collection is not None:
            for job in self.previous_schedule_jobs_collection.values():
                for operation in job.operations:
                    index = self.index_mapper.get_index_from_operation(operation)
                    if index is not None:
                        job_idx, op_idx = index
                        self.original_operation_starts[(job_idx, op_idx)] = operation.start

        # Active operations: block machines and delay jobs ---------------------------------------------
        if self.active_jobs_collection is not None:
            for job in self.active_jobs_collection.values():
                for operation in job.operations:
                    self.machines_fix_intervals.update_interval(
                        machine=operation.machine_name,
                        start = self.schedule_start,
                        end=operation.end
                    )
                    self.job_delays.update_delay(job_id=job.id, time_stamp=operation.end)

        # Operation-level constraints ------------------------------------------------------------------
        self._add_technological_operation_constraints()

        # Machine-level constraints (no overlap + fixed blocks from running ops) -----------------------
        for machine in self.machines:
            machine_intervals = []

            # Füge zu planende Operationen auf dieser Maschine hinzu
            for (_, _), (interval, machine_name) in self.intervals.items():
                if machine_name == machine:
                    machine_intervals.append(interval)

            # Füge evtl. blockierte Maschinenzeiten hinzu
            if machine in self.machines_fix_intervals:
                machine_fix_interval = self.machines_fix_intervals[machine]  # type: MachineFixInterval
                start = machine_fix_interval.start
                end = machine_fix_interval.end
                if start < end:
                    fixed_interval = self.model.NewIntervalVar(start, end - start, end, f"fixed_{machine}")
                    machine_intervals.append(fixed_interval)

            # NoOverlap für diese Maschine
            self.model.AddNoOverlap(machine_intervals)


    def build_model_for_jssp_lateness_with_start_deviation_minimization(
            self, previous_schedule_jobs_collection: Optional[LiveJobCollection] = None,
            active_jobs_collection: Optional[LiveJobCollection] = None,
            w_t: int = 1, w_e: int = 1, w_first: int = 1, main_pct: float = 0.5,
            duration_buffer_factor: float = 2.0):

        if self.model_completed:
            return "Model is already completed"

        self.previous_schedule_jobs_collection = previous_schedule_jobs_collection
        self.active_jobs_collection = active_jobs_collection

        self._build_reschedule_model()

        w_t, w_e, w_first = int(w_t), int(w_e), int(w_first)


        if previous_schedule_jobs_collection is None or previous_schedule_jobs_collection.count_operations() == 0:
            main_pct = 1.0

        print(f"{w_t=}, {w_e=}, {w_first=} {main_pct=}")

        main_pct_frac = Fraction(main_pct).limit_denominator(100)
        main_factor = main_pct_frac.numerator
        dev_factor = main_pct_frac.denominator - main_factor

        # Cost term containers -------------------------------------------------------------------------
        weighted_absolute_lateness_terms = []  # List of Job Lateness Terms (Tardiness + Earliness for last operations)
        first_op_terms = []  # List of 'First Earliness' Terms for First Operations of Jobs
        deviation_terms = []  # List of Deviation Penalty Terms (Difference from previous start times)

        # Additional operation-level constraints and objective terms -----------------------------------
        for (job_idx, op_idx), operation in self.index_mapper.items():
            start_var = self.start_times[(job_idx, op_idx)]
            end_var = self.end_times[(job_idx, op_idx)]

            if op_idx == 0:
                # Earliness of the "first" operation of a job ?????????????????????????????????????????????????????????????????
                first_op_latest_desired_start = int(
                    operation.job_deadline - operation.job.sum_duration * duration_buffer_factor)
                first_op_latest_desired_start = max(self.schedule_start, first_op_latest_desired_start)

                first_op_earliness = self.model.NewIntVar(0, self.horizon, f"first_op_earliness_{job_idx}")
                self.model.AddMaxEquality(first_op_earliness, [first_op_latest_desired_start - start_var, 0])
                term_first = self.model.NewIntVar(0, self.horizon * w_first, f"term_first_{job_idx}")
                self.model.Add(term_first == w_first * first_op_earliness)
                first_op_terms.append(term_first)

            # Lateness terms for the job (last operation)
            if operation.position_number == operation.job.last_operation_position_number:
                # Tardiness
                tardiness = self.model.NewIntVar(0, self.horizon, f"tardiness_{job_idx}")
                self.model.AddMaxEquality(tardiness, [end_var - operation.job_deadline, 0])
                term_tardiness = self.model.NewIntVar(0, self.horizon * w_t, f"term_tardiness_{job_idx}")
                self.model.Add(term_tardiness == w_t * tardiness)
                weighted_absolute_lateness_terms.append(term_tardiness)

                # Earliness
                earliness = self.model.NewIntVar(0, self.horizon, f"earliness_{job_idx}")
                self.model.AddMaxEquality(earliness, [operation.job_deadline - end_var, 0])
                term_earliness = self.model.NewIntVar(0, self.horizon * w_e, f"term_earliness_{job_idx}")
                self.model.Add(term_earliness == w_e * earliness)
                weighted_absolute_lateness_terms.append(term_earliness)

            # Deviation from original schedule
            if (job_idx, op_idx) in self.original_operation_starts.keys():
                deviation = self.model.NewIntVar(0, self.horizon, f"deviation_{job_idx}_{op_idx}")
                original_start = self.original_operation_starts[(job_idx, op_idx)]
                self.model.AddAbsEquality(deviation, start_var - original_start)
                deviation_terms.append(deviation)

        # Objective function ---------------------------------------------------------------------------

        # Weighted lateness = (tardiness + earliness) of last operation per job
        bound_lateness = (w_t + w_e) * self.horizon * len(self.jobs_collection.keys())
        absolute_lateness_part = self.model.NewIntVar(0, bound_lateness, "absolute_lateness_part")
        self.model.Add(absolute_lateness_part == sum(weighted_absolute_lateness_terms))

        # Weighted earliness of the first operations
        bound_first_op = w_first * self.horizon * len(self.jobs_collection.keys())
        first_op_earliness = self.model.NewIntVar(0, bound_first_op, "first_op_earliness")
        self.model.Add(first_op_earliness == sum(first_op_terms))

        # Total weighted lateness cost (scaled by main_factor)
        bound_lateness_target = main_factor * (bound_lateness + bound_first_op)
        target_scaled_lateness_part = self.model.NewIntVar(0, bound_lateness_target, "target_scaled_lateness_part")
        self.model.Add(target_scaled_lateness_part == main_factor * (absolute_lateness_part + first_op_earliness))

        # Weighted deviation cost (scaled by dev_factor)
        bound_deviation_target = dev_factor * self.horizon * len(deviation_terms)
        target_scaled_deviation_part = self.model.NewIntVar(0, bound_deviation_target,
                                                       "target_scaled_deviation_part")
        self.model.Add(target_scaled_deviation_part == dev_factor * sum(deviation_terms))

        # Final cost expression
        bound_total = bound_lateness_target + bound_deviation_target
        total_cost = self.model.NewIntVar(0, bound_total, "total_cost")
        self.model.Add(total_cost == target_scaled_lateness_part + target_scaled_deviation_part)
        self.model.Minimize(total_cost)

        self.model_completed = True

    def build_model_for_jssp_flowtime_with_start_deviation_minimization(
            self, previous_schedule_jobs_collection: Optional[LiveJobCollection] = None,
            active_jobs_collection: Optional[LiveJobCollection] = None, main_pct: float = 0.5,):

        self.previous_schedule_jobs_collection = previous_schedule_jobs_collection
        self.active_jobs_collection = active_jobs_collection

        self._build_reschedule_model()

        if previous_schedule_jobs_collection is None or previous_schedule_jobs_collection.count_operations() == 0:
            main_pct = 1.0

        main_pct_frac = Fraction(main_pct).limit_denominator(100)
        main_factor = main_pct_frac.numerator
        dev_factor = main_pct_frac.denominator - main_factor

        # Cost term containers -------------------------------------------------------------------------
        flowtime_terms = []
        deviation_terms = []

        # Additional operation-level constraints and objective terms -----------------------------------
        for (job_idx, op_idx), operation in self.index_mapper.items():
            start_var = self.start_times[(job_idx, op_idx)]
            end_var = self.end_times[(job_idx, op_idx)]

            # FlowTime (only last operation)
            if operation.position_number == operation.job.last_operation_position_number:
                flowtime = self.model.NewIntVar(0, self.horizon, f"flowtime_{job_idx}")
                self.model.Add(flowtime == end_var - operation.job_earliest_start)
                flowtime_terms.append(flowtime)

            # Deviation from original schedule
            if (job_idx, op_idx) in self.original_operation_starts.keys():
                deviation = self.model.NewIntVar(0, self.horizon, f"deviation_{job_idx}_{op_idx}")
                original_start = self.original_operation_starts[(job_idx, op_idx)]
                self.model.AddAbsEquality(deviation, start_var - original_start)
                deviation_terms.append(deviation)

        # Objective function ---------------------------------------------------------------------------
        bound_scaled_flow = main_factor * self.horizon * len(self.jobs_collection.keys())
        scaled_flow = self.model.NewIntVar(0, bound_scaled_flow, "scaled_flow")
        self.model.Add(scaled_flow == main_factor * sum(flowtime_terms))

        bound_scaled_dev = dev_factor * self.horizon * len(deviation_terms)
        scaled_dev = self.model.NewIntVar(0, bound_scaled_dev, "scaled_dev")
        self.model.Add(scaled_dev == dev_factor * sum(deviation_terms))

        total_cost = self.model.NewIntVar(0, bound_scaled_flow + bound_scaled_dev, "total_cost")
        self.model.Add(total_cost == scaled_flow + scaled_dev)
        self.model.Minimize(total_cost)

        self.model_completed = True


    def solve_model(
            self, print_log_search_progress: bool = False, time_limit: Optional[int] = None,
            gap_limit: float = 0.0, log_file: Optional[str] = None):

        if self.model_completed:

            self.solver.parameters.log_search_progress = print_log_search_progress
            self.solver.parameters.relative_gap_limit = gap_limit

            if time_limit is not None:
                self.solver.parameters.max_time_in_seconds = time_limit

            if log_file is not None:
                self.solver.parameters.log_search_progress = True
                with _redirect_cpp_logs(log_file):
                    self.solver_status = self.solver.Solve(self.model)
            else:
                self.solver_status = self.solver.Solve(self.model)
        else:
            print("Model is not completed")


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
                "number_of_operations_to_schedule": self.jobs_collection.count_operations(),
                "number_of_operations_with_previous_schedule": self.previous_schedule_jobs_collection.count_operations() if self.previous_schedule_jobs_collection else 0,
                "number_of_active_operation_to_consider": self.active_jobs_collection.count_operations() if self.active_jobs_collection else 0,
                "number_of_variables": len(model_proto.variables),
                "number_of_constraints": len(model_proto.constraints)
            }
            return model_info
        return "Model is not complete!"

    def get_solver_info(self):
        if self.solver_status:
            solver_info = {
                "status": self.solver.StatusName(self.solver_status),
                "objective_value": self.solver.ObjectiveValue() if self.solver_status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None,
                "best_objective_bound": self.solver.BestObjectiveBound(),
                "number_of_branches": self.solver.NumBranches(),
                "wall_time": round(self.solver.WallTime(), 2)
            }
            return solver_info
        return "Solver status is not available!"

    def get_experiment_log(self):
        if self.model_completed:
            experiment_log = {
                "model_info": self.get_model_info(),
                "solver_info": self.get_solver_info()
            }
            return experiment_log
        return "Model is not complete!"



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


