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

    @staticmethod
    def _build_basic_objects(jobs_collection: LiveJobCollection):

        # Model initialization and Helper objects ------------------------------------------------------
        model = cp_model.CpModel()
        index_mapper = OperationIndexMapper()
        start_times = StartTimes()
        end_times = EndTimes()
        intervals = Intervals()

        # Horizon (Worst-case upper bound)--------------------------------------------------------------
        total_duration = jobs_collection.get_total_duration()
        latest_deadline = jobs_collection.get_latest_deadline()
        horizon = latest_deadline + total_duration

        # Create Variables -----------------------------------------------------------------------------
        jobs_collection.sort_operations()
        jobs_collection.sort_jobs_by_arrival()

        for job_idx, job in enumerate(jobs_collection.values()):
            for op_idx, operation in enumerate(job.operations):
                suffix = f"{job_idx}_{op_idx}"
                start = model.NewIntVar(job.earliest_start, horizon, f"start_{suffix}")
                end = model.NewIntVar(job.earliest_start, horizon, f"end_{suffix}")

                interval = model.NewIntervalVar(start, operation.duration, end, f"interval_{suffix}")
                # interval = model.NewIntervalVar(start, operation.duration, start + operation.duration, f"interval_{suffix}")

                start_times[(job_idx, op_idx)] = start
                end_times[(job_idx, op_idx)] = end
                intervals[(job_idx, op_idx)] = (interval, operation.machine_name)
                index_mapper.add(job_idx, op_idx, operation)

        return model, index_mapper, start_times, end_times, intervals, horizon


    # Without previous schedule or delay due to active operations
    @classmethod
    def _build_basic_model(cls, jobs_collection: LiveJobCollection, schedule_start: int = 1440):

        # Model initialization -------------------------------------------------------------------------
        model, index_mapper, start_times, end_times, intervals, horizon = cls._build_basic_objects(jobs_collection)

        #  Machine-level constraints -------------------------------------------------------------------
        machines = jobs_collection.get_unique_machine_names()

        for machine in machines:
            machine_intervals = []

            # Add operation intervals for this machine
            for (_, _), (interval, machine_name) in intervals.items():
                if machine_name == machine:
                    machine_intervals.append(interval)

            # NoOverlap for this machine
            model.AddNoOverlap(machine_intervals)

        # Operation-level constraints ------------------------------------------------------------------
        for (job_idx, op_idx), operation in index_mapper.items():
            start_var = start_times[(job_idx, op_idx)]

            # 1. Technological constraint: earliest start of the first operation
            if op_idx == 0:
                min_start = max(operation.job_earliest_start, int(schedule_start))
                model.Add(start_var >= min_start)

            # 2. Technological constraint: operation order within the job
            if op_idx > 0:
                model.Add(start_var >= end_times[(job_idx, op_idx - 1)])

        return model, index_mapper, start_times, end_times, horizon

    @classmethod
    def build_makespan_model(cls, jobs_collection: LiveJobCollection, schedule_start: int = 1440):
        model, index_mapper, start_times, end_times, horizon = cls._build_basic_model(jobs_collection, schedule_start)
        makespan = model.NewIntVar(0, horizon, "makespan")
        for (job_idx, op_idx), operation in index_mapper.items():
            if operation.position_number == operation.job.last_operation_position_number:
                model.Add(makespan >= end_times[(job_idx, op_idx)])
        model.Minimize(makespan)
        return model, index_mapper, start_times, end_times

    # With previous schedule information and delay information (due to active operations)
    @classmethod
    def _build_model(
            cls, jobs_collection: LiveJobCollection,
            previous_schedule_jobs_collection: Optional[LiveJobCollection] = None,
            active_jobs_collection: Optional[LiveJobCollection] = None, schedule_start: int = 1440):

        # Model initialization and Helper objects ------------------------------------------------------
        model, index_mapper, start_times, end_times, intervals, horizon = cls._build_basic_objects(jobs_collection)

        # Objects for active operations
        machines_fix_intervals = MachineFixIntervalMap()
        job_delays = JobDelayMap()

        # Objects for previous schedule operations starts
        original_operation_starts = OriginalOperationStarts()

        # Machines -------------------------------------------------------------------------------------
        machines = jobs_collection.get_unique_machine_names()

        for machine in machines:
            machines_fix_intervals.add_interval(machine=machine, start=schedule_start, end=schedule_start)

        # Previous schedule: extract start times for deviation penalties -------------------------------
        if previous_schedule_jobs_collection is not None:
            for job in previous_schedule_jobs_collection.values():
                for operation in job.operations:
                    index = index_mapper.get_index_from_operation(operation)
                    if index is not None:
                        job_idx, op_idx = index
                        original_operation_starts[(job_idx, op_idx)] = operation.start

        # Active operations: block machines and delay jobs ---------------------------------------------
        if active_jobs_collection is not None:
            for job in active_jobs_collection.values():
                for operation in job.operations:
                    machines_fix_intervals.update_interval(machine=operation.machine_name, end=operation.end)
                    job_delays.update_delay(job_id=job.id, time_stamp=operation.end)

        # Machine-level constraints (no overlap + fixed blocks from running ops) -----------------------
        for machine in machines:
            machine_intervals = []

            # Füge zu planende Operationen auf dieser Maschine hinzu
            for (_, _), (interval, machine_name) in intervals.items():
                if machine_name == machine:
                    machine_intervals.append(interval)

            # Füge evtl. blockierte Maschinenzeiten hinzu
            if machine in machines_fix_intervals:
                machine_fix_interval = machines_fix_intervals[machine]  # type: MachineFixInterval
                start = machine_fix_interval.start
                end = machine_fix_interval.end
                if start < end:
                    fixed_interval = model.NewIntervalVar(start, end - start, end, f"fixed_{machine}")
                    machine_intervals.append(fixed_interval)

            # NoOverlap für diese Maschine
            model.AddNoOverlap(machine_intervals)

        # Operation-level constraints and objective terms ----------------------------------------------
        for (job_idx, op_idx), operation in index_mapper.items():
            start_var = start_times[(job_idx, op_idx)]

            # 1.Technological constraint: earliest start of the first operation
            if op_idx == 0:
                # Earliest_start of the "first" operation of a job
                min_start = max(operation.job_earliest_start, int(schedule_start))
                if operation.job.id in job_delays:
                    earliest_start = job_delays.get_delay(operation.job.id).earliest_start
                    min_start = max(min_start, earliest_start)
                model.Add(start_var >= min_start)

            # 2. Technological constraint: operation order within the job
            if op_idx > 0:
                model.Add(start_var >= end_times[(job_idx, op_idx - 1)])

        return model, index_mapper, start_times, end_times, horizon, job_delays, original_operation_starts


    @classmethod
    def build_model_for_jssp_lateness_with_start_deviation_minimization(
            cls, jobs_collection: LiveJobCollection,
            previous_schedule_jobs_collection: Optional[LiveJobCollection] = None,
            active_jobs_collection: Optional[LiveJobCollection] = None, schedule_start: int = 1440,
            w_t: int = 1, w_e: int = 1, w_first: int = 1, main_pct: float = 0.5,
            duration_buffer_factor: float = 2.0):

        model, index_mapper, start_times, end_times, horizon, job_delays, original_operation_starts = cls._build_model(
            jobs_collection=jobs_collection,
            previous_schedule_jobs_collection=previous_schedule_jobs_collection,
            active_jobs_collection=active_jobs_collection,
            schedule_start=schedule_start,
        )

        w_t, w_e, w_first = int(w_t), int(w_e), int(w_first)

        if previous_schedule_jobs_collection is None or previous_schedule_jobs_collection.count_operations() == 0:
            main_pct = 1.0

        main_pct_frac = Fraction(main_pct).limit_denominator(100)
        main_factor = main_pct_frac.numerator
        dev_factor = main_pct_frac.denominator - main_factor

        # Cost term containers -------------------------------------------------------------------------
        weighted_absolute_lateness_terms = []  # List of Job Lateness Terms (Tardiness + Earliness for last operations)
        first_op_terms = []  # List of 'First Earliness' Terms for First Operations of Jobs
        deviation_terms = []  # List of Deviation Penalty Terms (Difference from previous start times)

        # Operation-level constraints and objective terms ----------------------------------------------

        for (job_idx, op_idx), operation in index_mapper.items():
            start_var = start_times[(job_idx, op_idx)]
            end_var = start_times[(job_idx, op_idx)]

            if op_idx == 0:
                # Earliness of the "first" operation of a job ?????????????????????????????????????????????????????????????????
                first_op_latest_desired_start = int(
                    operation.job_deadline - operation.job.sum_duration * duration_buffer_factor)
                first_op_latest_desired_start = max(schedule_start, first_op_latest_desired_start)

                first_op_earliness = model.NewIntVar(0, horizon, f"first_op_earliness_{job_idx}")
                model.AddMaxEquality(first_op_earliness, [first_op_latest_desired_start - start_var, 0])
                term_first = model.NewIntVar(0, horizon * w_first, f"term_first_{job_idx}")
                model.Add(term_first == w_first * first_op_earliness)
                first_op_terms.append(term_first)

            # Lateness terms for the job (last operation)
            if operation.position_number == operation.job.last_operation_position_number:
                # Tardiness
                tardiness = model.NewIntVar(0, horizon, f"tardiness_{job_idx}")
                model.AddMaxEquality(tardiness, [end_var - operation.job_deadline, 0])
                term_tardiness = model.NewIntVar(0, horizon * w_t, f"term_tardiness_{job_idx}")
                model.Add(term_tardiness == w_t * tardiness)
                weighted_absolute_lateness_terms.append(term_tardiness)

                # Earliness
                earliness = model.NewIntVar(0, horizon, f"earliness_{job_idx}")
                model.AddMaxEquality(earliness, [operation.job_deadline - end_var, 0])
                term_earliness = model.NewIntVar(0, horizon * w_e, f"term_earliness_{job_idx}")
                model.Add(term_earliness == w_e * earliness)
                weighted_absolute_lateness_terms.append(term_earliness)

            # Deviation from original schedule
            if (job_idx, op_idx) in original_operation_starts.keys():
                deviation = model.NewIntVar(0, horizon, f"deviation_{job_idx}_{op_idx}")
                original_start = original_operation_starts[(job_idx, op_idx)]
                model.AddAbsEquality(deviation, start_var - original_start)
                deviation_terms.append(deviation)

        # Objective function ---------------------------------------------------------------------------

        # Weighted lateness = (tardiness + earliness) of last operation per job
        bound_lateness = (w_t + w_e) * horizon * len(jobs_collection.keys())
        absolute_lateness_part = model.NewIntVar(0, bound_lateness, "absolute_lateness_part")
        model.Add(absolute_lateness_part == sum(weighted_absolute_lateness_terms))

        # Weighted earliness of the first operations
        bound_first_op = w_first * horizon * len(jobs_collection.keys())
        first_op_earliness = model.NewIntVar(0, bound_first_op, "first_op_earliness")
        model.Add(first_op_earliness == sum(first_op_terms))

        # Total weighted lateness cost (scaled by main_factor)
        bound_lateness_target = main_factor * (bound_lateness + bound_first_op)
        target_scaled_lateness_part = model.NewIntVar(0, bound_lateness_target, "target_scaled_lateness_part")
        model.Add(target_scaled_lateness_part == main_factor * (absolute_lateness_part + first_op_earliness))

        # Weighted deviation cost (scaled by dev_factor)
        bound_deviation_target = dev_factor * horizon * len(deviation_terms)
        target_scaled_deviation_part = model.NewIntVar(0, bound_deviation_target,
                                                       "target_scaled_deviation_part")
        model.Add(target_scaled_deviation_part == dev_factor * sum(deviation_terms))

        # Final cost expression
        bound_total = bound_lateness_target + bound_deviation_target
        total_cost = model.NewIntVar(0, bound_total, "total_cost")
        model.Add(total_cost == target_scaled_lateness_part + target_scaled_deviation_part)
        model.Minimize(total_cost)

        return model, index_mapper, start_times, end_times

    @classmethod
    def build_model_for_jssp_flowtime_with_start_deviation_minimization(
            cls, jobs_collection: LiveJobCollection,
            previous_schedule_jobs_collection: Optional[LiveJobCollection] = None,
            active_jobs_collection: Optional[LiveJobCollection] = None, schedule_start: int = 1440,
            main_pct: float = 0.5,):

        model, index_mapper, start_times, end_times, horizon, job_delays, original_operation_starts = cls._build_model(
            jobs_collection=jobs_collection,
            previous_schedule_jobs_collection=previous_schedule_jobs_collection,
            active_jobs_collection=active_jobs_collection,
            schedule_start=schedule_start,
        )

        if previous_schedule_jobs_collection is None or previous_schedule_jobs_collection.count_operations() == 0:
            main_pct = 1.0

        main_pct_frac = Fraction(main_pct).limit_denominator(100)
        main_factor = main_pct_frac.numerator
        dev_factor = main_pct_frac.denominator - main_factor

        # Cost term containers -------------------------------------------------------------------------
        flowtime_terms = []
        deviation_terms = []

        # Operation-level constraints and objective terms ----------------------------------------------
        for (job_idx, op_idx), operation in index_mapper.items():
            start_var = start_times[(job_idx, op_idx)]
            end_var = start_times[(job_idx, op_idx)]

            # FlowTime (only last operation)
            if operation.position_number == operation.job.last_operation_position_number:
                flowtime = model.NewIntVar(0, horizon, f"flowtime_{job_idx}")
                model.Add(flowtime == end_var - operation.job_earliest_start)
                flowtime_terms.append(flowtime)

            # Deviation from original schedule
            if (job_idx, op_idx) in original_operation_starts.keys():
                deviation = model.NewIntVar(0, horizon, f"deviation_{job_idx}_{op_idx}")
                original_start = original_operation_starts[(job_idx, op_idx)]
                model.AddAbsEquality(deviation, start_var - original_start)
                deviation_terms.append(deviation)

        # Objective function ---------------------------------------------------------------------------
        bound_scaled_flow = main_factor * horizon * len(jobs_collection.keys())
        scaled_flow = model.NewIntVar(0, bound_scaled_flow, "scaled_flow")
        model.Add(scaled_flow == main_factor * sum(flowtime_terms))

        bound_scaled_dev = dev_factor * horizon * len(deviation_terms)
        scaled_dev = model.NewIntVar(0, bound_scaled_dev, "scaled_dev")
        model.Add(scaled_dev == dev_factor * sum(deviation_terms))

        total_cost = model.NewIntVar(0, bound_scaled_flow + bound_scaled_dev, "total_cost")
        model.Add(total_cost == scaled_flow + scaled_dev)
        model.Minimize(total_cost)

        return model, index_mapper, start_times, end_times


    @staticmethod
    def solve_model(
            model: cp_model.CpModel, print_log_search_progress: bool = False, time_limit: Optional[int] = None,
            gap_limit: float = 0.0, log_file: Optional[str] = None):

        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = print_log_search_progress
        solver.parameters.relative_gap_limit = gap_limit

        if time_limit is not None:
            solver.parameters.max_time_in_seconds = time_limit

        if log_file is not None:
            with _redirect_cpp_logs(log_file):
                status = solver.Solve(model)
        else:
            status = solver.Solve(model)

        return solver, status


    @staticmethod
    def get_schedule(
            index_mapper: OperationIndexMapper,
            start_times: StartTimes, end_times: EndTimes, solver: cp_model.CpSolver):
        pass
        schedule_job_collection = LiveJobCollection()

        for (job_idx, op_idx), operation in index_mapper.items():
            start = solver.Value(start_times[(job_idx, op_idx)])
            end = solver.Value(end_times[(job_idx, op_idx)])

            schedule_job_collection.add_operation_instance(
                op=operation,
                new_start=start,
                new_end=end
            )

        return schedule_job_collection


    @staticmethod
    def get_model_info(jobs_collection: LiveJobCollection, previous_schedule_jobs_collection: LiveJobCollection,
                       active_jobs_collection: LiveJobCollection, model: cp_model):
        model_proto = model.Proto()
        model_info = {
            "number_of_operations_to_schedule": jobs_collection.count_operations(),
            "number_of_operations_with_previous_schedule": previous_schedule_jobs_collection.count_operations() if previous_schedule_jobs_collection else 0,
            "number_of_active_operation_to_consider": active_jobs_collection.count_operations() if active_jobs_collection else 0,
            "number_of_variables": len(model_proto.variables),
            "number_of_constraints": len(model_proto.constraints)
        }
        return model_info

    @staticmethod
    def get_solver_info(solver: cp_model.CpSolver, status: CpSolverStatus):
        solver_info = {
            "status": solver.StatusName(status),
            "objective_value": solver.ObjectiveValue() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None,
            "best_objective_bound": solver.BestObjectiveBound(),
            "number_of_branches": solver.NumBranches(),
            "wall_time": solver.WallTime()
        }
        return solver_info

    @classmethod
    def solve_jssp_lateness_with_start_deviation_minimization(
            cls, jobs_collection: LiveJobCollection,
            previous_schedule_jobs_collection: Optional[LiveJobCollection] = None,
            active_jobs_collection: Optional[LiveJobCollection] = None,
            w_t: int = 5, w_e: int = 1, w_first: int = 1, main_pct: float = 0.5,
            duration_buffer_factor: float = 2.0, schedule_start: int = 1440,
            solver_print_log_search_progress: bool = False, solver_time_limit: Optional[int] = 3600,
            solver_relative_gap_limit: float = 0.0, log_file: Optional[str] = None):

        model, index_mapper, start_times, end_times = cls.build_model_for_jssp_lateness_with_start_deviation_minimization(
            jobs_collection=jobs_collection,
            previous_schedule_jobs_collection=previous_schedule_jobs_collection,
            active_jobs_collection=active_jobs_collection, schedule_start=schedule_start,
            w_t=w_t, w_e=w_e, w_first=w_first,
            main_pct=main_pct, duration_buffer_factor=duration_buffer_factor,
        )

        solver, status = cls.solve_model(
            model=model,
            print_log_search_progress=solver_print_log_search_progress,
            time_limit=solver_time_limit,
            gap_limit=solver_relative_gap_limit,
            log_file=log_file
        )

        schedule_job_collection = cls.get_schedule(
            index_mapper=index_mapper,
            start_times=start_times,
            end_times=end_times,
            solver=solver
        )

        experiment_log = {
            "model_info": cls.get_model_info(
                jobs_collection=jobs_collection,
                previous_schedule_jobs_collection=previous_schedule_jobs_collection,
                active_jobs_collection=active_jobs_collection,
                model=model
            ),
            "solver_info": cls.get_solver_info(
                solver=solver,
                status=status
            )
        }
        return schedule_job_collection, experiment_log



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


