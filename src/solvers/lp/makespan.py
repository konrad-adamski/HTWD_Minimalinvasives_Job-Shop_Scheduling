import pulp
from typing import Dict, List, Tuple, Optional, Literal

from src.solvers.lp.problem_builder import add_machine_conflict_constraints, add_technological_constraints, \
    add_makespan_definition
from src.solvers.lp.problem_solver import solve_lp_problem_and_extract_schedule


def solve_jssp_makespan(
    job_ops: Dict[str, List[Tuple[int, str, int]]],
    job_earliest_starts: Optional[Dict[str, int]] = None,
    solver_type: Literal["CBC", "HiGHS"] = "CBC",
    var_cat: Literal["Continuous", "Integer"] = "Continuous",
    msg: bool = False,
    solver_time_limit: int = 3600,
    solver_relative_gap_limit: float = 0.0,
    log_file: Optional[str] = None
) -> List[Tuple[str, int, str, int, int, int]]:
    """
    Solves the job-shop scheduling problem to minimize makespan using a MILP model (via PuLP).
    Respects technological order, machine constraints, and optional earliest job start times.

    :param job_ops: Dictionary mapping each job to a list of operations (operation_index, machine, duration).
    :type job_ops: dict[str, list[tuple[int, str, int]]]
    :param job_earliest_starts: Optional dictionary with the earliest start time per job.
                                If None, all jobs are assumed to be available at time 0.
    :type job_earliest_starts: dict[str, int] or None
    :param solver_type: MILP solver to use. Must be one of 'CBC' or 'HiGHS'.
                        If 'HiGHS' is selected, the HiGHS solver must be installed!
    :type solver_type: Literal["CBC", "HiGHS"]
    :param var_cat: Type of LP variables to use. Must be either 'Continuous' (default) or 'Integer'.
    :type var_cat: Literal["Continuous", "Integer"]
    :param time_limit: Optional time limit for the solver in seconds.
    :type time_limit: int or None
    :param solver_args: Additional solver arguments passed to the PuLP solver command.
    :return: List of scheduled operations as tuples (job, operation, machine, start, duration, end).
    :rtype: list[tuple[str, int, str, int, int, int]]
    """

    if job_earliest_starts is None:
        job_earliest_starts = {job: 0 for job in job_ops}

    machines = {machine for ops in job_ops.values() for _, machine, _ in ops}

    # === Build MILP Model ==
    problem = pulp.LpProblem("JSSP_Makespan_Model", pulp.LpMinimize)

    starts = {
        (job, o): pulp.LpVariable(f"start_{job}_{o}", lowBound=job_earliest_starts[job], cat=var_cat)
        for job, ops in job_ops.items()
        for o in range(len(ops))
    }

    makespan = pulp.LpVariable("makespan", lowBound=0, cat=var_cat)
    problem += makespan

    # === Add Constraints ===
    add_technological_constraints(problem, starts, job_ops)
    add_makespan_definition(problem, starts, job_ops, makespan)

    # === Add machine disjunctive constraints using Big-M ===
    # Prevents overlap on machines; Big-M = max release + total processing
    total_duration = sum(d for ops in job_ops.values() for _, _, d in ops)
    max_arrival = max(job_earliest_starts.values())
    big_m = max_arrival + total_duration
    add_machine_conflict_constraints(prob=problem, starts=starts, job_ops=job_ops, machines=machines,big_m=big_m)

    # === Solve and Extract Schedule ===
    schedule, solver_info = solve_lp_problem_and_extract_schedule(
        problem=problem, starts=starts, job_ops=job_ops, solver_type=solver_type, time_limit=solver_time_limit,
        gap_limit=solver_relative_gap_limit, msg = msg, log_file=log_file
    )

    # === Solver Info ==
    print("\nSolver Information:")
    for key, value in solver_info.items():
        print(f"  {key.replace('_', ' ').capitalize():25}: {value}")
    return schedule
