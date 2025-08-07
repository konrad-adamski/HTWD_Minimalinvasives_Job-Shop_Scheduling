import time

from typing import Dict, List, Tuple, Optional, Set



def solve(
    job_ops: Dict[str, List[Tuple[int, str, int]]],
    job_earliest_starts: Optional[Dict[str, int]] = None
) -> List[Tuple[str, int, str, int, int, int]]:
    """
    Schedules operations based on a given job_ops model using the First-Come First-Served (FCFS) heuristic.
    Optionally considers earliest start times per job.

    :param job_ops: Dictionary mapping each job to a list of operations (operation_index, machine, duration).
    :type job_ops: dict[str, list[tuple[int, str, int]]]
    :param job_earliest_starts: Optional dictionary with the earliest start time per job.
                                If None, all jobs are assumed to be available at time 0.
    :type job_earliest_starts: dict[str, int] or None
    :return: List of scheduled operations in the form (job, operation, machine, start, duration, end).
    :rtype: list[tuple[str, int, str, int, int, int]]
    """
    start_time = time.time()

    if job_earliest_starts is None:
        job_earliest_starts = {job: 0 for job in job_ops}

    machines = get_machines_from_job_ops(job_ops)

    job_ready = job_earliest_starts.copy()
    machine_ready = {m: 0 for m in machines}
    pointer = {job: 0 for job in job_ops}
    total_ops = sum(len(ops) for ops in job_ops.values())

    schedule = []

    while total_ops > 0:
        best = None
        for job in sorted(job_ops):
            p = pointer[job]
            if p >= len(job_ops[job]):
                continue

            op, machine, dur = job_ops[job][p]
            earliest = max(job_ready[job], machine_ready[machine])

            if best is None or earliest < best[1] or (
                earliest == best[1] and job_earliest_starts[job] < job_earliest_starts[best[0]]
            ):
                best = (job, earliest, dur, machine, op)

        job, start, dur, machine, op = best
        end = start + dur
        schedule.append((job, op, machine, start, dur, end))

        job_ready[job] = end
        machine_ready[machine] = end
        pointer[job] += 1
        total_ops -= 1

    # Logging
    makespan = max(end for *_, end in schedule)
    solving_duration = time.time() - start_time
    print("\nPlanungsinformationen (FCFS):")
    print(f"  Anzahl Operationen  : {len(schedule)}")
    print(f"  Makespan            : {makespan}")
    print(f"  Laufzeit            : ~{solving_duration:.4f} Sekunden")

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

