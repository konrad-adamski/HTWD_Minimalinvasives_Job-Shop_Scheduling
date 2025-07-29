import pulp

from typing import Dict, List, Tuple, Set

def add_machine_conflict_constraints(
                prob: pulp.LpProblem, starts: Dict[Tuple[str, int], pulp.LpVariable],
                job_ops: Dict[str, List[Tuple[int, str, int]]], machines: Set[str],
                big_m: float) -> None:
    """
    Adds disjunctive machine conflict constraints to a PuLP MILP model for job-shop scheduling.

    Ensures that no two operations assigned to the same machine overlap in time.
    Introduces binary disjunction variables to enforce mutual exclusion using Big-M logic.

    :param prob: The PuLP problem instance to which constraints are added.
    :param starts: Dictionary mapping (job, operation index) to PuLP start time variables.
    :param job_ops: Dictionary mapping each job to its sequence of operations
                    as tuples (operation_index, machine, duration).
    :param machines: Set of all machine identifiers involved in the schedule.
    :param big_m: Big-M constant used in disjunctive time separation constraints.
    """
    for m in machines:
        ops_on_m = [
            (job, o, ops[o][2])
            for job, ops in job_ops.items()
            for o in range(len(ops))
            if ops[o][1] == m
        ]
        for i in range(len(ops_on_m)):
            j1, o1, d1 = ops_on_m[i]
            for j2, o2, d2 in ops_on_m[i + 1:]:
                if j1 == j2:
                    continue
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                prob += starts[(j1, o1)] + d1 <= starts[(j2, o2)] + big_m * (1 - y)
                prob += starts[(j2, o2)] + d2 <= starts[(j1, o1)] + big_m * y



def add_technological_constraints(problem, starts, job_ops):
    for job, ops in job_ops.items():
        for o in range(1, len(ops)):
            d_prev = ops[o - 1][2]
            problem += starts[(job, o)] >= starts[(job, o - 1)] + d_prev


def add_makespan_definition(problem, starts, job_ops, makespan):
    for job, ops in job_ops.items():
        last_op_idx = len(ops) - 1
        d_last = ops[last_op_idx][2]
        problem += makespan >= starts[(job, last_op_idx)] + d_last



