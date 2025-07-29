from typing import List, Tuple, Dict, Set

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

