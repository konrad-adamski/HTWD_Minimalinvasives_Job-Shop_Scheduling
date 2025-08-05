import collections

from fractions import Fraction
from ortools.sat.python import cp_model
from typing import Dict, Tuple, List, Optional, Literal, Any, Union

from src.classes.Collection import JobMixCollection
from src.classes.orm_models import Job, JobTemplate, JobOperation
from src.solvers.cp.model_classes import MachineFixInterval, MachineFixIntervalMap, JobDelayMap, OperationIndexMapper
from src.solvers.cp.model_solver import solve_cp_model_and_extract_schedule


def solve_jssp_lateness_with_deviation_minimization(
        jobs_collection: JobMixCollection,
        previous_schedule_jobs_collection: Optional[JobMixCollection] = None,
        active_jobs_collection: Optional[JobMixCollection] = None,
        w_t: int = 5, w_e: int = 1, w_first: int = 1, main_pct: float = 0.5,
        duration_buffer_factor: float = 2.0, schedule_start: int = 1440,
        deviation_type: Literal["start", "order_on_machine"] = "start", msg: bool = False,
        solver_time_limit: Optional[int] = 3600, solver_relative_gap_limit: float = 0.0,
        log_file: Optional[str] = None) -> Tuple[JobMixCollection, Dict[str, Any]]:

    # 1. === Model initialization and weight preprocessing ===
    model = cp_model.CpModel()
    w_t, w_e, w_first = int(w_t), int(w_e), int(w_first)

    if previous_schedule_jobs_collection is None or previous_schedule_jobs_collection.count_operations() == 0:
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

    starts, ends, intervals = {}, {}, {}

    index_mapper = OperationIndexMapper()

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
            index_mapper.add(job_idx, op_idx, operation)


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

    if previous_schedule_jobs_collection is not None:
        for job in previous_schedule_jobs_collection.values():
            for operation in job.operations:
                index = index_mapper.get_index_from_operation(operation)
                if index is not None:
                    job_idx, op_idx = index
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

    if active_jobs_collection is not None:
        for job in active_jobs_collection.values():
            for operation in job.operations:
                    machines_fix_intervals.update_interval(machine=operation.machine, end=operation.end)
                    job_delays.update_delay(job_id=job.id, time_stamp=operation.end)

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
            if operation.job.id in job_delays:
                earliest_start = job_delays.get_delay(operation.job.id).earliest_start
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
    bound_lateness = (w_t + w_e) * horizon * len(jobs_collection.keys())
    absolute_lateness_part = model.NewIntVar(0, bound_lateness, "absolute_lateness_part")
    model.Add(absolute_lateness_part == sum(weighted_absolute_lateness_terms))

    # Weighted earliness of the first operations
    bound_first_op = w_first * horizon * len(jobs_collection.keys())
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
    schedule_job_collection, solver_info = solve_cp_model_and_extract_schedule(
        model=model, index_mapper=index_mapper, starts=starts, ends=ends,
        msg=msg, time_limit=solver_time_limit, gap_limit=solver_relative_gap_limit, log_file=log_file)

    # 11. === Experiment Logging ===
    model_proto = model.Proto()
    experiment_log = {
        "experiment_info": {
            "total_number_of_operations": jobs_collection.count_operations(),
            "number_of_operations_with_previous_schedule": previous_schedule_jobs_collection.count_operations() if previous_schedule_jobs_collection else 0,
            "number_of_active_operation_to_consider": active_jobs_collection.count_operations()  if active_jobs_collection else 0,
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

    return schedule_job_collection, experiment_log

# Wrappers ------------------------------------------------------------------------------------------------------------

def solve_jssp_lateness_with_start_deviation_minimization(
        jobs_collection: JobMixCollection,
        previous_schedule_jobs_collection: Optional[JobMixCollection] = None,
        active_jobs_collection: Optional[JobMixCollection] = None,
        w_t: int = 5, w_e: int = 1, w_first: int = 1, main_pct: float = 0.5,
        duration_buffer_factor: float = 2.0, schedule_start: int = 1440, msg: bool = False,
        solver_time_limit: Optional[int] = 3600, solver_relative_gap_limit: float = 0.0,
        log_file: Optional[str] = None) -> Tuple[JobMixCollection, Dict[str, Any]]:

    return solve_jssp_lateness_with_deviation_minimization(
        jobs_collection=jobs_collection, previous_schedule_jobs_collection=previous_schedule_jobs_collection,
        active_jobs_collection=active_jobs_collection, w_t=w_t, w_e=w_e, w_first=w_first, main_pct=main_pct,
        duration_buffer_factor=duration_buffer_factor, schedule_start=schedule_start, deviation_type="start",
        msg=msg, solver_time_limit=solver_time_limit, solver_relative_gap_limit=solver_relative_gap_limit,
        log_file = log_file
    )

def solve_jssp_lateness_with_order_deviation_minimization(
        jobs_collection: JobMixCollection,
        previous_schedule_jobs_collection: Optional[JobMixCollection] = None,
        active_jobs_collection: Optional[JobMixCollection] = None,
        w_t: int = 5, w_e: int = 1, w_first: int = 1, main_pct: float = 0.5,
        duration_buffer_factor: float = 2.0, schedule_start: int = 1440, msg: bool = False,
        solver_time_limit: Optional[int] = 3600, solver_relative_gap_limit: float = 0.0,
        log_file: Optional[str] = None) -> Tuple[JobMixCollection, Dict[str, Any]]:
    return solve_jssp_lateness_with_deviation_minimization(
        jobs_collection=jobs_collection, previous_schedule_jobs_collection=previous_schedule_jobs_collection,
        active_jobs_collection=active_jobs_collection, w_t=w_t, w_e=w_e, w_first=w_first, main_pct=main_pct,
        duration_buffer_factor=duration_buffer_factor, schedule_start=schedule_start, deviation_type="order_on_machine",
        msg=msg, solver_time_limit=solver_time_limit, solver_relative_gap_limit=solver_relative_gap_limit,
        log_file=log_file
    )



def solve_jssp_tardiness_minimization(
        jobs_collection: JobMixCollection,
        previous_schedule_jobs_collection: Optional[JobMixCollection] = None,
        active_jobs_collection: Optional[JobMixCollection] = None,
        main_pct: float = 1.0, duration_buffer_factor: float = 2.0, schedule_start: int = 1440,
        deviation_type: Literal["start", "order_on_machine"] = "start", msg: bool = False,
        solver_time_limit: Optional[int] = 3600, solver_relative_gap_limit: float = 0.0,
        log_file: Optional[str] = None) -> Tuple[JobMixCollection, Dict[str, Any]]:

    return solve_jssp_lateness_with_deviation_minimization(
        jobs_collection=jobs_collection, previous_schedule_jobs_collection=previous_schedule_jobs_collection,
        active_jobs_collection=active_jobs_collection,
        w_t=1, w_e=0, w_first=0, main_pct=main_pct, duration_buffer_factor=duration_buffer_factor,
        schedule_start=schedule_start, deviation_type=deviation_type, msg=msg, solver_time_limit=solver_time_limit,
        solver_relative_gap_limit=solver_relative_gap_limit, log_file = log_file
    )