from typing import Dict, Tuple, List, Any, Optional, Literal
import pulp
import time


def solve_lp_problem_and_extract_schedule(
    problem: pulp.LpProblem,
    starts: Dict[Tuple[str, int], pulp.LpVariable],
    job_ops: Dict[str, List[Tuple[int, str, int]]],
    solver_type: Literal["CBC", "HiGHS"] = "CBC",
    var_cat: Literal["Continuous", "Integer"] = "Continuous",
    time_limit: int = 3600,
    gap_limit: float = 0.0,
    msg: bool = False,
    log_file: Optional[str] = None,
) -> Tuple[List[Tuple[str, int, str, int, int, int]], Dict[str, Any]]:
    """
    Solves a PuLP MILP problem and extracts the resulting schedule if feasible.

    :param problem: The PuLP problem instance to be solved.
    :param starts: Dictionary of start time variables indexed by (job, operation_index).
    :param job_ops: Dictionary mapping each job to its list of operations.
                    Each operation is a tuple (op_id, machine, duration).
    :param solver_type: Solver to use, either "CBC" or "HiGHS".
    :param var_cat: Variable category: "Continuous" or "Integer".
    :param time_limit: Maximum solver time in seconds.
    :param gap_limit: Acceptable relative MIP gap.
    :param msg: If True, enable solver log output.
    :param log_file: Optional path to file for redirecting solver output.
    :return: Tuple of (scheduled operations list, solver info dict).
    """
    start_timer = time.time()

    solver_args = {
        "timeLimit": time_limit,
        "gapRel": gap_limit,
        "msg": msg
    }

    if log_file is not None:
        solver_args["logPath"] = log_file

    if solver_type == "HiGHS":
        cmd = pulp.HiGHS_CMD(**solver_args)
    elif solver_type == "CBC":
        cmd = pulp.PULP_CBC_CMD(**solver_args)
    else:
        raise ValueError("solver_type must be either 'CBC' or 'HiGHS'.")

    problem.solve(cmd)

    schedule = []
    for job, ops in job_ops.items():
        for o, (op_id, machine, duration) in enumerate(ops):
            start_val = starts[(job, o)].varValue
            end_val = start_val + duration
            schedule.append((job, op_id, machine, round(start_val, 2), duration, round(end_val, 2)))

    solver_info = {
        "status": pulp.LpStatus[problem.status],
        "objective_value": pulp.value(problem.objective),
        "runtime": round(time.time() - start_timer, 2),
        "num_variables": len(problem.variables()),
        "num_constraints": len(problem.constraints)
    }

    return schedule, solver_info