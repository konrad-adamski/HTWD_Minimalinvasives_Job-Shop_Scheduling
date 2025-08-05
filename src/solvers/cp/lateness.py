import collections
import math

from fractions import Fraction
from ortools.sat.python import cp_model
from typing import Dict, Tuple, List, Optional, Literal, Any, Union

from src.classes.Collection import JobMixCollection
from src.classes.orm_models import Job, JobTemplate, JobOperation
from src.solvers.cp.model_builder import build_cp_variables, extract_active_ops_info, \
    add_machine_constraints, compute_job_total_durations, get_last_operation_index, \
    add_order_on_machines_deviation_terms, extract_original_start_times_and_machine_order, \
    add_kendall_tau_deviation_terms
from src.solvers.cp.model_classes import MachineFixInterval, MachineFixIntervalMap, JobDelayMap
from src.solvers.cp.model_solver import solve_cp_model_and_extract_schedule


def solve_jssp_lateness_with_deviation_minimization(
        jobs_collection: JobMixCollection,
        previous_schedule_jobs_collection: Optional[JobMixCollection] = None,
        active_jobs_collection: Optional[JobMixCollection] = None,
        w_t: int = 5, w_e: int = 1, w_first: int = 1, main_pct: float = 0.5,
        duration_buffer_factor: float = 2.0, schedule_start: int = 1440,
        deviation_type: Literal["start", "order_on_machine"] = "start", msg: bool = False,
        solver_time_limit: Optional[int] = 3600, solver_relative_gap_limit: float = 0.0,
        log_file: Optional[str] = None) -> Tuple[List[Tuple[str, int, str, int, int, int]], Dict[str, Any]]:

    # 1. === Model initialization and weight preprocessing ===
    model = cp_model.CpModel()
    w_t, w_e, w_first = int(w_t), int(w_e), int(w_first)

    if not previous_schedule_jobs_collection:   #  or is empty!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! implementieren!!!!!!!!!!!!!!!!!!!!!!!!!
        main_pct = 1.0

    main_pct_frac = Fraction(main_pct).limit_denominator(100)
    main_factor = main_pct_frac.numerator
    dev_factor = main_pct_frac.denominator - main_factor

    # Machines --------------------------------------------------------------------------------------------------------
    machines = jobs_collection.get_unique_machines()

    machines_fix_intervals = MachineFixIntervalMap()

    for machine in machines:
        machines_fix_intervals.add_interval(machine=machine, start=schedule_start, end=schedule_start)

    # Horizon ---------------------------------------------------------------------------------------------------------
    total_duration = jobs_collection.get_total_duration()
    latest_deadline = jobs_collection.get_latest_deadline()
    horizon = latest_deadline + total_duration

    # Create Variables ----------------------------------------------------------------------------------------------
    jobs_collection.sort_operations()
    jobs_collection.sort_jobs_by_arrival()

    starts, ends, intervals, index_mapper = {}, {}, {}, {}
    index_mapper = {}
    for job_idx, job in enumerate(jobs_collection.values()):
        for op_idx, operation in enumerate(job.operations):
            operation: JobOperation
            suffix = f"{job_idx}_{op_idx}"
            start = model.NewIntVar(job.earliest_start, horizon, f"start_{suffix}")
            end = model.NewIntVar(job.earliest_start, horizon, f"end_{suffix}")

            interval = model.NewIntervalVar(start, operation.duration, end, f"interval_{suffix}")
            # interval = model.NewIntervalVar(start, operation.duration, start + operation.duration, f"interval_{suffix}")

            starts[(job_idx, op_idx)] = start
            ends[(job_idx, op_idx)] = end
            intervals[(job_idx, op_idx)] = (interval, operation.machine)
            index_mapper[(job_idx, op_idx)] = operation


    operation_to_index = {operation: index for index, operation in index_mapper.items()}



    # 4. === Preparation: job durations, last operations, cost term containers ===
    #job_total_duration = compute_job_total_durations(operations)
    #last_op_index = get_last_operation_index(operations)

    # --- term containers ---
    weighted_absolute_lateness_terms = []   # List of Job Lateness Terms (Tardiness + Earliness for last operations)
    first_op_terms = []                     # List of 'First Earliness' Terms for First Operations of Jobs
    deviation_terms = []                    # List of Deviation Penalty Terms (Difference from previous start times)

    # 5. === Previous schedule: extract start times and orders on machines for deviation penalties ===
    #original_start, original_machine_orders = extract_original_start_times_and_machine_order(
    #    previous_schedule,
    #    operations
    #)

    original_operation_starts = {}
    original_machine_orders = collections.defaultdict(list)

    if previous_schedule_jobs_collection:  # prüfe auf None und Leere !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for job in previous_schedule_jobs_collection.values():
            for operation in job.operations:
                if operation in operation_to_index:
                    job_idx, op_idx = operation_to_index[operation]
                    original_operation_starts[(job_idx, op_idx)] = operation.start
                    original_machine_orders[operation.machine].append((operation.start, job_idx, op_idx))

        for machine in original_machine_orders:
            original_machine_orders[machine].sort()
            original_machine_orders[machine] = [(job_idx, op_idx) for _, job_idx, op_idx in original_machine_orders[machine]]



    # 6. === Active operations: block machines and delay jobs ===
    #machines_delays, job_ops_delays = extract_active_ops_info(active_ops, schedule_start)

    #machines_delays: Dict[str, Tuple[int, int]] = {}
    #job_ops_delays: Dict[str, int] = {}

    job_delays = JobDelayMap()

    if active_jobs_collection is not None: # oder leer --------------------------------------------------------
        for job in active_jobs_collection.values():
            for operation in job.operations:
                    machines_fix_intervals.update_interval(machine=operation.machine, end=operation.end)
                    job_delays.update_delay(job=job, time_stamp=operation.end)

    # 7. === Machine-level constraints (no overlap + fixed blocks from running ops) ===

    #add_machine_constraints(model, machines, intervals, machines_delays)

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


    #if deviation_type == "order_on_machine" and original_machine_orders:                                                    # TODO
    #    pass
    #    #deviation_terms += add_order_on_machines_deviation_terms(model, original_machine_orders, operations, starts)
    #    deviation_terms += add_kendall_tau_deviation_terms(model, original_machine_orders, operations)

    # 8. === Operation-level constraints and objective terms ===

    for (job_idx, op_idx), operation in index_mapper.items():
        start_var = starts[(job_idx, op_idx)]
        end_var = ends[(job_idx, op_idx)]

        if op_idx == 0:
            # Earliest_start of the "first" operation of a job
            min_start = max(operation.job_earliest_start, int(schedule_start))
            if operation.job in job_delays:
                earliest_start = job_delays.get_delay(operation.job).earliest_start
                min_start = max(min_start, earliest_start)
            model.Add(start_var >= min_start)

            # Earliness of the "first" operation of a job
            first_op_latest_desired_start = int(operation.job_deadline - operation.job.sum_duration * duration_buffer_factor)
            first_op_latest_desired_start = max(schedule_start, first_op_latest_desired_start)

            first_op_earliness = model.NewIntVar(0, horizon, f"first_op_earliness_{job_idx}")
            model.AddMaxEquality(first_op_earliness, [first_op_latest_desired_start - start_var, 0])
            term_first = model.NewIntVar(0, horizon * w_first, f"term_first_{job_idx}")
            model.Add(term_first == w_first * first_op_earliness)
            first_op_terms.append(term_first)


        # Technological constraint (precedence within the job)
        if op_idx > 0:
            model.Add(start_var >= ends[(job_idx, op_idx - 1)])


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
            model.AddMaxEquality(earliness, [ operation.job_deadline - end_var, 0])
            term_earliness = model.NewIntVar(0, horizon * w_e, f"term_earliness_{job_idx}")
            model.Add(term_earliness == w_e * earliness)
            weighted_absolute_lateness_terms.append(term_earliness)



        # Deviation from original schedule
        if deviation_type == "start" and (job_idx, op_idx) in original_operation_starts.keys():
            deviation = model.NewIntVar(0, horizon, f"deviation_{job_idx}_{op_idx}")
            original_start = original_operation_starts[(job_idx, op_idx)]
            model.AddAbsEquality(deviation, start_var - original_start)
            deviation_terms.append(deviation)



    # 9. === Objective function ===

    # Weighted lateness = (tardiness + earliness) of last operation per job
    bound_lateness = (w_t + w_e) * horizon * len(jobs)
    absolute_lateness_part = model.NewIntVar(0, bound_lateness, "absolute_lateness_part")
    model.Add(absolute_lateness_part == sum(weighted_absolute_lateness_terms))

    # Weighted earliness of the first operations
    bound_first_op = w_first * horizon * len(jobs)
    first_op_earliness = model.NewIntVar(0, bound_first_op , "first_op_earliness")
    model.Add(first_op_earliness == sum(first_op_terms))

    # Total weighted lateness cost (scaled by main_factor)
    bound_lateness_target = main_factor * (bound_lateness + bound_first_op)
    target_scaled_lateness_part = model.NewIntVar(0, bound_lateness_target, "target_scaled_lateness_part")
    model.Add(target_scaled_lateness_part == main_factor * (absolute_lateness_part + first_op_earliness))

    # Weighted deviation cost (scaled by dev_factor)
    bound_deviation_target = dev_factor * horizon * len(deviation_terms)
    target_scaled_deviation_part = model.NewIntVar(0, bound_deviation_target, "target_scaled_deviation_part")
    model.Add(target_scaled_deviation_part == dev_factor * sum(deviation_terms))

    # Final cost expression
    bound_total = bound_lateness_target + bound_deviation_target
    total_cost = model.NewIntVar(0, bound_total, "total_cost")
    model.Add(total_cost == target_scaled_lateness_part + target_scaled_deviation_part)
    model.Minimize(total_cost)

    # 10. === Solve and extract solution ===
    schedule, solver_info = solve_cp_model_and_extract_schedule(
        model=model, operations=operations, starts=starts, ends=ends,
        msg=msg, time_limit=solver_time_limit, gap_limit=solver_relative_gap_limit, log_file=log_file)

    # 11. === Experiment Logging ===
    model_proto = model.Proto()

    experiment_log = {
        "experiment_info": {
            "total_number_of_operations": len(operations),
            "number_of_operations_with_previous_schedule": len(original_start),
            "number_of_active_operation_to_consider": len(active_ops) if active_ops else 0,
            "schedule_start": schedule_start,
        },
        "experiment_config": {
            "main_pct": float(main_pct),
            "w_t": w_t,
            "w_e": w_e,
            "w_first": w_first,
            "deviation_type": deviation_type,
            "solver_time_limit": solver_time_limit,
            "solver_relative_gap_limit": solver_relative_gap_limit,
        },
        "model_info": {
            "number_of_variables": len(model_proto.variables),
            "number_of_constraints": len(model_proto.constraints),
            "number_of_deviation_terms": len(deviation_terms),
        },
        "solver_info": solver_info
    }

    return schedule, experiment_log

# Wrappers ------------------------------------------------------------------------------------------------------------

def solve_jssp_lateness_with_start_deviation_minimization(
        job_ops: Dict[str, List[Tuple[int, str, int]]], times_dict: Dict[str, Tuple[int, int]],
        previous_schedule: Optional[List[Tuple[str, int, str, int, int, int]]] = None,
        active_ops: Optional[List[Tuple[str, int, str, int, int, int]]] = None,
        w_t: int = 5, w_e: int = 1, w_first: int = 1, main_pct: float = 0.5,
        duration_buffer_factor: float = 2.0, schedule_start: int = 1440, msg: bool = False,
        solver_time_limit: Optional[int] = 3600, solver_relative_gap_limit: float = 0.0,
        log_file: Optional[str] = None) -> Tuple[List[Tuple[str, int, str, int, int, int]], Dict[str, Any]]:
    """
    Solves a JSSP minimizing lateness and start-time deviation.

    :param job_ops: Job operations as {job: [(op_id, machine, duration), ...]}.
    :param times_dict: Mapping from job to (earliest_start, deadline).
    :param previous_schedule: Optional prior plan [(job, op_id, machine, start, duration, end)].
    :param active_ops: Optional running ops (same format as previous_schedule).
    :param w_t: Weight for tardiness.
    :param w_e: Weight for earliness.
    :param w_first: Weight for early first operation starts.
    :param main_pct: Portion of cost focused on lateness (0.0–1.0).
    :param duration_buffer_factor: Factor for relaxed first op start.
    :param schedule_start: Earliest time for new operations.
    :param msg: Enable solver log output.
    :param solver_time_limit: Max solver time (in seconds).
    :param solver_relative_gap_limit: Allowed MIP gap.
    :param log_file: Optional path to file for redirecting solver output.

    :return: A tuple of:
         - List of scheduled operations: (job, op_id, machine, start, duration, end)
         - Dictionary with experiment log data
    """
    return solve_jssp_lateness_with_deviation_minimization(
        job_ops=job_ops, times_dict=times_dict, previous_schedule=previous_schedule, active_ops=active_ops,
        w_t=w_t, w_e=w_e, w_first=w_first, main_pct=main_pct, duration_buffer_factor=duration_buffer_factor,
        schedule_start=schedule_start, deviation_type="start", msg=msg, solver_time_limit=solver_time_limit,
        solver_relative_gap_limit=solver_relative_gap_limit, log_file = log_file
    )

def solve_jssp_lateness_with_order_deviation_minimization(
        job_ops: Dict[str, List[Tuple[int, str, int]]], times_dict: Dict[str, Tuple[int, int]],
        previous_schedule: Optional[List[Tuple[str, int, str, int, int, int]]] = None,
        active_ops: Optional[List[Tuple[str, int, str, int, int, int]]] = None,
        w_t: int = 5, w_e: int = 1, w_first: int = 1, main_pct: float = 0.5,
        duration_buffer_factor: float = 2.0, schedule_start: int = 1440, msg: bool = False,
        solver_time_limit: Optional[int] = 3600, solver_relative_gap_limit: float = 0.0,
        log_file: Optional[str] = None) -> Tuple[List[Tuple[str, int, str, int, int, int]], Dict[str, Any]]:
    """
    Solves a JSSP minimizing lateness and deviation from original machine order.

    :param job_ops: Job operations as {job: [(op_id, machine, duration), ...]}.
    :param times_dict: Mapping from job to (earliest_start, deadline).
    :param previous_schedule: Optional prior plan [(job, op_id, machine, start, duration, end)].
    :param active_ops: Optional running ops (same format as previous_schedule).
    :param w_t: Weight for tardiness.
    :param w_e: Weight for earliness.
    :param w_first: Weight for early first operation starts.
    :param main_pct: Portion of cost focused on lateness (0.0–1.0).
    :param duration_buffer_factor: Factor for relaxed first op start.
    :param schedule_start: Earliest time for new operations.
    :param msg: Enable solver log output.
    :param solver_time_limit: Max solver time (in seconds).
    :param solver_relative_gap_limit: Allowed MIP gap.
    :param log_file: Optional path to file for redirecting solver output.

    :return: A tuple of:
         - List of scheduled operations: (job, op_id, machine, start, duration, end)
         - Dictionary with experiment log data
    """
    return solve_jssp_lateness_with_deviation_minimization(
        job_ops=job_ops, times_dict=times_dict, previous_schedule=previous_schedule, active_ops=active_ops,
        w_t=w_t, w_e=w_e, w_first=w_first, main_pct=main_pct, duration_buffer_factor=duration_buffer_factor,
        schedule_start=schedule_start, deviation_type="order_on_machine", msg=msg, solver_time_limit=solver_time_limit,
        solver_relative_gap_limit=solver_relative_gap_limit, log_file = log_file
    )


def solve_jssp_tardiness_minimization(
        job_ops: Dict[str, List[Tuple[int, str, int]]], times_dict: Dict[str, Tuple[int, int]],
        previous_schedule: Optional[List[Tuple[str, int, str, int, int, int]]] = None,
        active_ops: Optional[List[Tuple[str, int, str, int, int, int]]] = None,
        main_pct: float = 1.0, duration_buffer_factor: float = 2.0, schedule_start: int = 1440,
        deviation_type: Literal["start", "order_on_machine"] = "start", msg: bool = False,
        solver_time_limit: Optional[int] = 3600, solver_relative_gap_limit: float = 0.0,
        log_file: Optional[str] = None) -> Tuple[List[Tuple[str, int, str, int, int, int]], Dict[str, Any]]:
    """
    Solves a JSSP minimizing tardiness, optionally with deviation penalties.

    :param job_ops: Job operations as {job: [(op_id, machine, duration), ...]}.
    :param times_dict: Mapping from job to (earliest_start, deadline).
    :param previous_schedule: Optional prior plan [(job, op_id, machine, start, duration, end)].
    :param active_ops: Optional running ops (same format as previous_schedule).
    :param main_pct: Portion of objective weight on tardiness (0.0–1.0).
    :param duration_buffer_factor: Factor for relaxed start target of first op.
    :param schedule_start: Lower bound for all operation start times.
    :param deviation_type: Deviation type: "start" or "order_on_machine".
    :param msg: Enable solver log output.
    :param solver_time_limit: Max solver time in seconds.
    :param solver_relative_gap_limit: Allowed MIP gap.
    :param log_file: Optional path to file for redirecting solver output.

    :return: A tuple of:
         - List of scheduled operations: (job, op_id, machine, start, duration, end)
         - Dictionary with experiment log data
    """
    return solve_jssp_lateness_with_deviation_minimization(
        job_ops=job_ops, times_dict=times_dict, previous_schedule=previous_schedule, active_ops=active_ops,
        w_t=1, w_e=0, w_first=0, main_pct=main_pct, duration_buffer_factor=duration_buffer_factor,
        schedule_start=schedule_start, deviation_type=deviation_type, msg=msg, solver_time_limit=solver_time_limit,
        solver_relative_gap_limit=solver_relative_gap_limit, log_file = log_file
    )