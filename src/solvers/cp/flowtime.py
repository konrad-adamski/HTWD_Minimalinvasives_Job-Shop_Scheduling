import math

from fractions import Fraction
from ortools.sat.python import cp_model
from typing import Dict, List, Optional, Tuple, Literal

from src.solvers.cp.model_builder import build_cp_variables, extract_active_ops_info, \
    add_machine_constraints, get_last_operation_index, add_order_on_machines_deviation_terms, \
    extract_original_start_times_and_machine_order
from src.solvers.cp.model_solver import solve_cp_model_and_extract_schedule


def solve_jssp_flowtime_with_deviation_minimization(
    job_ops: Dict[str, List[Tuple[int, str, int]]],
    earliest_start: Dict[str, int],
    previous_schedule: Optional[List[Tuple[str, int, str, int, int, int]]] = None,
    active_ops: Optional[List[Tuple[str, int, str, int, int, int]]] = None,
    main_pct: float = 0.5,
    schedule_start: int = 1440,
    deviation_type: Literal["start", "order_on_machine"] = "start",
    msg: bool = False,
    solver_time_limit: int = 3600,
    solver_relative_gap_limit: float = 0.0,
    log_file: Optional[str] = None) -> List[Tuple[str, int, str, int, int, int]]:
    """
    Solve a Job-Shop Scheduling Problem (JSSP) using CP-SAT with soft objective terms for total flow time
    and deviation from a previous schedule.

    This solver supports:
    a) Flow time minimization (sum of job end times minus arrival times).
    b) Deviation minimization from previously planned start times or machine order.
    c) Integration of active operations, which block machines and delay job continuation.

    :param job_ops: Dictionary mapping each job to a list of operations.
                    Each operation is a tuple (operation_id, machine, duration).
    :type job_ops: Dict[str, List[Tuple[int, str, int]]]
    :param earliest_start: Dictionary mapping each job to its earliest possible start time.
    :type earliest_start: Dict[str, int]
    :param previous_schedule: Optional list of previously scheduled operations.
                              Format: (job, op_id, machine, start, duration, end).
    :type previous_schedule: Optional[List[Tuple[str, int, str, int, int, int]]]
    :param active_ops: Optional list of currently running operations (e.g., from a simulation snapshot).
                       Format: (job, op_id, machine, start, duration, end).
    :type active_ops: Optional[List[Tuple[str, int, str, int, int, int]]]
    :param main_pct: Fraction (0.0–1.0) of total objective weight allocated to flow time.
                     The remaining weight is applied to deviation penalties.
    :type main_pct: float
    :param schedule_start: Lower bound for any newly scheduled operation (rescheduling start point).
    :type schedule_start: int
    :param deviation_type: Specifies the type of deviation penalty to use in the objective.
                           - "start": penalizes absolute differences between planned and previous start times.
                           - "order_on_machine": penalizes inversions in operation order on the same machine compared to the previous schedule.
    :type deviation_type: Literal["start", "order_on_machine"]
    :param msg: If True, enable solver log output.
    :type msg: bool
    :param solver_time_limit: Maximum allowed solving time (in seconds).
    :type solver_time_limit: int
    :param solver_relative_gap_limit: Allowed relative gap between best and proven bound.
    :type solver_relative_gap_limit: float
    :param log_file: Optional path to file for redirecting solver output.
    :type log_file: Optional[str]

    :return: List of scheduled operations as (job, op_id, machine, start, duration, end).
    :rtype: List[Tuple[str, int, str, int, int, int]]
    """

    # 1. === Model initialization and weight preprocessing ===
    model = cp_model.CpModel()

    if not previous_schedule:
        main_pct = 1.0

    main_frac = Fraction(main_pct).limit_denominator(100)
    main_factor = main_frac.numerator
    dev_factor = main_frac.denominator - main_factor

    # 2. === Preprocessing: arrivals, machines, planning horizon ===
    jobs = list(job_ops.keys())
    machines = {m for ops in job_ops.values() for _, m, _ in ops}

    # Worst-case upper bound for time horizon
    total_duration = sum(d for ops in job_ops.values() for (_, _, d) in ops)
    horizon = max(max(earliest_start.values()), schedule_start) + total_duration

    # 3. === Create variables ===
    starts, ends, intervals, operations = build_cp_variables(model, job_ops, earliest_start, horizon)
    last_op_index = get_last_operation_index(operations)

    # 4. === Preparation: Cost terms ===
    flowtime_terms = []
    deviation_terms = []

    # 5. === Previous schedule: extract start times and orders on machines for deviation penalties ===
    original_start, original_machine_orders = extract_original_start_times_and_machine_order(
        previous_schedule,
        operations
    )

    # 6. === Active operations: block machines and delay jobs ===
    machines_delays, job_ops_delays = extract_active_ops_info(active_ops, schedule_start)

    # 7. === Machine constraints ===
    add_machine_constraints(model, machines, intervals, machines_delays)

    if deviation_type == "order_on_machine" and original_machine_orders:
        deviation_terms += add_order_on_machines_deviation_terms(model, original_machine_orders, operations, starts)

    # 8. === Operation-level constraints and cost assignments ===
    for job_idx, job, op_idx, op_id, machine, duration in operations:
        start_var = starts[(job_idx, op_idx)]
        end_var = ends[(job_idx, op_idx)]

        # Respect the earliest start: arrival time and machine delay (based on active operations)
        min_start = max(earliest_start[job], schedule_start)
        if job in job_ops_delays:
            min_start = max(min_start, int(math.ceil(job_ops_delays[job])))
        model.Add(start_var >= min_start)

        # Technological constraint (precedence within the job)
        if op_idx > 0:
            model.Add(start_var >= ends[(job_idx, op_idx - 1)])

        # Deviation from original schedule
        if deviation_type == "start":
            key = (job, op_id)
            if key in original_start:
                diff = model.NewIntVar(-horizon, horizon, f"diff_{job_idx}_{op_idx}")
                dev = model.NewIntVar(0, horizon, f"dev_{job_idx}_{op_idx}")
                model.Add(diff == start_var - original_start[key])
                model.AddAbsEquality(dev, diff)
                deviation_terms.append(dev)

        # FlowTime (only last operation)
        if op_idx == last_op_index[job]:
            arrival = earliest_start[job]
            flowtime = model.NewIntVar(0, horizon, f"flowtime_{job}")
            model.Add(flowtime == end_var - arrival)
            flowtime_terms.append(flowtime)

    # 9. === Objective function ===
    bound_scaled_flow = main_factor * horizon * len(jobs)
    scaled_flow = model.NewIntVar(0, bound_scaled_flow, "scaled_flow")
    model.Add(scaled_flow == main_factor * sum(flowtime_terms))

    bound_scaled_dev = dev_factor * horizon * len(deviation_terms)
    scaled_dev = model.NewIntVar(0, bound_scaled_dev, "scaled_dev")
    model.Add(scaled_dev == dev_factor * sum(deviation_terms))

    total_cost = model.NewIntVar(0, bound_scaled_flow + bound_scaled_dev, "total_cost")
    model.Add(total_cost == scaled_flow + scaled_dev)
    model.Minimize(total_cost)


    # 10. === Model-Log summary ===
    print("Model Information")
    model_proto = model.Proto()
    print(f"  Number of variables       : {len(model_proto.variables)}")
    print(f"  Number of constraints     : {len(model_proto.constraints)}")
    print(f"  Deviation terms (IntVars) : {len(deviation_terms)}")

    # 11. === Solve and extract solution ===
    schedule, solver_info = solve_cp_model_and_extract_schedule(
        model=model, operations=operations, starts=starts, ends=ends,
        msg=msg, time_limit=solver_time_limit, gap_limit=solver_relative_gap_limit, log_file=log_file)

    print("\nSolver Information:")
    for key, value in solver_info.items():
        print(f"  {key.replace('_', ' ').capitalize():25}: {value}")
    return schedule

# Wrappers ------------------------------------------------------------------------------------------------------------

def solve_jssp_flowtime_minimization(
    job_ops: Dict[str, List[Tuple[int, str, int]]],
    earliest_start: Dict[str, int],
    schedule_start: int = 0,
    msg: bool = False,
    solver_time_limit: int = 3600,
    solver_relative_gap_limit: float = 0.0,
    log_file: Optional[str] = None
) -> List[Tuple[str, int, str, int, int, int]]:
    """
    Solve JSSP with flow time minimization only (no deviation or active ops).

    :param job_ops: Dictionary of job operations (op_id, machine, duration).
    :param earliest_start: Dictionary of earliest start times per job.
    :param schedule_start: Time from which scheduling begins.
    :param msg: Enable solver logs.
    :param solver_time_limit: Max time allowed for solver.
    :param solver_relative_gap_limit: Allowed relative gap.
    :param log_file: Optional log file for solver output.

    :return: List of scheduled operations.
    """
    schedule = solve_jssp_flowtime_with_deviation_minimization(
        job_ops=job_ops,
        earliest_start=earliest_start,
        previous_schedule=None,
        active_ops=None,
        main_pct=1.0,
        schedule_start=schedule_start,
        deviation_type="start",  # ignored anyway when main_pct = 1
        msg=msg,
        solver_time_limit=solver_time_limit,
        solver_relative_gap_limit=solver_relative_gap_limit,
        log_file=log_file
    )
    return schedule