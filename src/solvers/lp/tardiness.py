import pulp

from typing import Dict, List, Tuple, Optional, Literal
from src.solvers.lp.problem_builder import add_machine_conflict_constraints, add_technological_constraints
from src.solvers.lp.problem_solver import solve_lp_problem_and_extract_schedule


def solve_jssp_sum_tardiness_minimization(
        job_ops: Dict[str, List[Tuple[int, str, int]]],
        times_dict: Dict[str, Tuple[int, int]],
        solver_type: Literal["CBC", "HiGHS"] = "CBC",
        var_cat: Literal["Continuous", "Integer"] = "Continuous",
        msg: bool = False,
        solver_time_limit: int = 3600,
        solver_relative_gap_limit: float = 0.0,
        log_file: Optional[str] = None
    ) -> List[Tuple[str, int, str, int, int, int]]:
    """
    Solves the job-shop scheduling problem using a MILP model (via PuLP) to minimize the total tardiness across all jobs.
    The model accounts for the technological order of operations within each job, ensures
    that no two operations on the same machine overlap, and respects the earliest start time (arrival) for each job.
    Tardiness is defined as the amount of time a job ends after its specified deadline (if any).
    The objective is to minimize the sum of all job-specific tardiness values.

    :param job_ops: Dictionary mapping each job to a list of operations (operation_id, machine, duration).
    :type job_ops: Dict[str, List[Tuple[int, str, int]]]

    :param times_dict: Dictionary mapping each job to a tuple of (arrival_time, deadline_time).
    :type times_dict: Dict[str, Tuple[int, int]]

    :param solver_type: MILP solver to use. Must be either 'CBC' or 'HiGHS'. Requires HiGHS installed if selected.
    :type solver_type: Literal["CBC", "HiGHS"]

    :param var_cat: Variable category for LP model: 'Continuous' or 'Integer'.
    :type var_cat: Literal["Continuous", "Integer"]

    :param msg: If True, prints solver logs and timing info.
    :type msg: bool

    :param solver_time_limit: Time limit for solver execution (in seconds).
    :type solver_time_limit: int

    :param solver_relative_gap_limit: Acceptable relative MIP gap.
    :type solver_relative_gap_limit: float

    :param log_file: Optional file path to store solver output log.
    :type log_file: Optional[str]

    :return: List of scheduled operations in format: (job, operation_id, machine, start, duration, end)
    :rtype: List[Tuple[str, int, str, int, int, int]]
    """
    jobs = list(job_ops.keys())
    job_earliest_start = {job: times_dict[job][0] for job in jobs}
    deadline = {job: times_dict[job][1] for job in jobs}
    machines = get_machines_from_job_ops(job_ops)

    # === 1. Modell & Variablen ===
    problem = pulp.LpProblem("JSSP_Tardiness_Minimization", pulp.LpMinimize)

    starts = {}
    job_ends = {}
    job_tards = {}

    for job in jobs:
        ops = job_ops[job]
        for o, (op_id, machine, duration) in enumerate(ops):
            starts[(job, o)] = pulp.LpVariable(f"start_{job}_{op_id}", lowBound=0, cat=var_cat)

        job_ends[job] = pulp.LpVariable(f"end_{job}", lowBound=0, cat=var_cat)
        job_tards[job] = pulp.LpVariable(f"tard_{job}", lowBound=0, cat=var_cat)

    # === 2. Constraints ===
    add_technological_constraints(problem, starts, job_ops)

    for job in jobs:
        # Earliest allowed start time for the first operation
        problem += starts[(job, 0)] >= job_earliest_start[job], f"earliest_start_{job}"

        # Define job end time: start of last operation + its duration
        last_op_index = len(job_ops[job]) - 1
        last_op = job_ops[job][last_op_index]
        problem += job_ends[job] == starts[(job, last_op_index)] + last_op[2], f"endtime_def_{job}"

        # Tardiness constraint: Tardiness ≥ End time − Deadline
        problem += job_tards[job] >= job_ends[job] - deadline[job], f"tardiness_def_{job}"


    # === Add machine disjunctive constraints using Big-M ===
    # Prevents overlap on machines; Big-M = max release + total processing
    total_duration = sum(d for ops in job_ops.values() for _, _, d in ops)
    max_earliest_start = max(job_earliest_start.values())
    big_m = max_earliest_start + total_duration
    add_machine_conflict_constraints(prob=problem, starts=starts, job_ops=job_ops, machines=machines, big_m=big_m)

    # === Target ===
    problem += pulp.lpSum(job_tards.values())

    # === Solve and Extract Schedule ===
    schedule, solver_info = solve_lp_problem_and_extract_schedule(
        problem=problem, starts=starts, job_ops=job_ops, solver_type=solver_type, time_limit=solver_time_limit,
        gap_limit=solver_relative_gap_limit, msg=msg, log_file=log_file
    )

    # === Solver Info ==
    print("\nSolver Information:")
    for key, value in solver_info.items():
        print(f"  {key.replace('_', ' ').capitalize():25}: {value}")
    return schedule



def get_machines_from_job_ops(job_ops: Dict[str, List[Tuple[int, str, int]]]) -> Set[str]:
    """
    Extracts the set of unique machines used across all operations in a job_ops model.

    :param job_ops: Dictionary mapping each job to a list of operations (operation_index, machine, duration).
    :type job_ops: dict[str, list[tuple[int, str, int]]]
    :return: Set of unique machine identifiers used in the job shop model.
    :rtype: set[str]
    """
    machines = {machine for ops in job_ops.values() for _, machine, _ in ops}
    return machines

