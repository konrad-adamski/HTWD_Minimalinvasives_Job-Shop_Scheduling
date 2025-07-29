import collections
from typing import Dict, List, Tuple, Optional, Any
from ortools.sat.python import cp_model


def build_cp_variables(
        model: cp_model.CpModel, job_ops: Dict[str, List[Tuple[int, str, int]]],
        job_earliest_starts: Dict[str, int], horizon: int
) -> Tuple[Dict[Tuple[int, int], cp_model.IntVar],
           Dict[Tuple[int, int], cp_model.IntVar],
           Dict[Tuple[int, int], Tuple[cp_model.IntervalVar, str]],
           List[Tuple[int, str, int, int, str, int]]]:
    """
        Builds CP-SAT variables for a job-shop scheduling model.

        This function generates start, end, and interval variables for each job operation using
        internal integer indices (`job_idx`, `op_idx`) to ensure efficient and consistent mapping.
        The operations are sorted by `op_id` to enforce the correct technological sequence within each job.

        :param model: The OR-Tools CP model instance.
        :type model: cp_model.CpModel
        :param job_ops: Dictionary mapping job names to a list of operations.
                       Each operation is a tuple: (op_id, machine, duration).
        :type job_ops: Dict[str, List[Tuple[int, str, int]]]
        :param job_earliest_starts: Dictionary with the earliest start time per job.
        :type job_earliest_starts: Dict[str, int]
        :param horizon: The upper bound on the scheduling time horizon.
        :type horizon: int

        :returns: A tuple with:
                  - **starts**: Mapping from (job_idx, op_idx) to start time variable (IntVar).
                  - **ends**: Mapping from (job_idx, op_idx) to end time variable (IntVar).
                  - **intervals**: Mapping from (job_idx, op_idx) to (IntervalVar, machine).
                  - **operations**: List of operation metadata tuples:
                    (job_idx, job_name, op_idx, op_id, machine, duration).
        :rtype: Tuple[
                    Dict[Tuple[int, int], cp_model.IntVar],
                    Dict[Tuple[int, int], cp_model.IntVar],
                    Dict[Tuple[int, int], Tuple[cp_model.IntervalVar, str]],
                    List[Tuple[int, str, int, int, str, int]]
                ]
        """
    starts, ends, intervals, op_data = {}, {}, {}, []
    jobs = sorted(job_ops.keys())

    for job_idx, job_name in enumerate(jobs):

        # Sort job_ops[job_name] by op_id to ensure op_idx follows the technological order
        sorted_ops = sorted(job_ops[job_name], key=lambda x: x[0])

        for op_idx, (op_id, machine, duration) in enumerate(sorted_ops):
            suffix = f"{job_idx}_{op_idx}"
            est = job_earliest_starts[job_name]
            start = model.NewIntVar(est, horizon, f"start_{suffix}")
            end = model.NewIntVar(est, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, duration, end, f"interval_{suffix}")

            starts[(job_idx, op_idx)] = start
            ends[(job_idx, op_idx)] = end
            intervals[(job_idx, op_idx)] = (interval, machine)
            op_data.append((job_idx, job_name, op_idx, op_id, machine, duration))

    return starts, ends, intervals, op_data


def extract_active_ops_info(
    active_ops: Optional[List[Tuple[str, int, str, int, int, int]]],
    schedule_start: int
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, int]]:
    """
    Extracts resource and job delay information from already active operations.

    Used to ensure that:
    - machines are blocked until the end of running operations,
    - and successor operations in a job do not start before the last active one.

    :param active_ops: List of tuples representing active or finished operations.
                       Each tuple is (job, op_id, machine, start, duration, end).
    :param schedule_start: Start time of the rescheduling window. Only operations
                           ending after or at this time are considered.

    :return:
        - machines_delays: Dict mapping each machine to a blocking interval (start, end)
        - job_ops_delays: Dict mapping each job to the latest end time of its active operations
    """
    machines_delays: Dict[str, Tuple[int, int]] = {}
    job_ops_delays: Dict[str, int] = {}

    if active_ops:
        for job, _, machine, _, _, end in active_ops:
            if end >= schedule_start:
                if machine not in machines_delays or end > machines_delays[machine][1]:
                    machines_delays[machine] = (schedule_start, end)
            if job not in job_ops_delays or end > job_ops_delays[job]:
                job_ops_delays[job] = end

    return machines_delays, job_ops_delays

def compute_job_total_durations(
    operations: List[Tuple[int, str, int, int, str, int]]
) -> Dict[str, int]:
    """
    Computes the total duration of all operations for each job.

    :param operations: List of tuples (job_idx, job_name, op_idx, op_id, machine, duration)
    :return: Dict mapping job_name to total processing duration
    """
    job_total_duration = {}
    for _, job, _, _, _, duration in operations:
        job_total_duration[job] = job_total_duration.get(job, 0) + duration
    return job_total_duration

def get_last_operation_index(
    operations: List[Tuple[int, str, int, int, str, int]]
) -> Dict[str, int]:
    """
    Determines the highest operation index (op_idx) for each job.

    :param operations: List of tuples (job_idx, job_name, op_idx, op_id, machine, duration)
    :return: Dict mapping job_name to last op_idx
    """
    last_op_index = {}
    for _, job, op_idx, _, _, _ in operations:
        last_op_index[job] = max(op_idx, last_op_index.get(job, -1))
    return last_op_index


def extract_original_start_times(
        previous_schedule: Optional[List[Tuple[str, int, str, int, int, int]]],
        operations: List[Tuple[int, str, int, int, str, int]]) -> Dict[Tuple[str, int], int]:
    """
    Extracts original start times from a previous schedule,
    restricted to operations currently present in the model.

    :param previous_schedule: List of (job, op_id, machine, start, duration, end)
    :param operations: List of (job_idx, job_name, op_idx, op_id, machine, duration)
    :return: Dict mapping (job, op_id) to original start time
    """
    original_start, _ = extract_original_start_times_and_machine_order(previous_schedule, operations)
    return original_start


def extract_original_start_times_and_machine_order(
        previous_schedule: Optional[List[Tuple[str, int, str, int, int, int]]],
        operations: List[Tuple[int, str, int, int, str, int]]
) -> Tuple[Dict[Tuple[str, int], int], Dict[str, List[Tuple[str, int]]]]:
    """
    Extracts original start times and machine-wise operation order from a previous schedule,
    restricted to operations currently present in the model.

    :param previous_schedule: List of (job, op_id, machine, start, duration, end)
    :param operations: List of (job_idx, job_name, op_idx, op_id, machine, duration)
    :return: Tuple containing a dict mapping (job, op_id) to original start time
             and a dict mapping each machine to a list of (job, op_id) sorted by start time
    """
    original_operation_starts = {}
    original_machine_orders = collections.defaultdict(list)

    if previous_schedule:
        valid_keys = {(job, op_id) for _, job, _, op_id, _, _ in operations}
        for job, op_id, machine, start, duration, end in previous_schedule:
            key = (job, op_id)
            if key in valid_keys:
                original_operation_starts[key] = start
                original_machine_orders[machine].append((start, job, op_id))

        for m in original_machine_orders:
            original_machine_orders[m].sort()   # sort by start (first element of tuple)
            original_machine_orders[m] = [(job, op_id) for _, job, op_id in original_machine_orders[m]]

    return original_operation_starts, original_machine_orders



def add_machine_constraints(
    model: cp_model.CpModel,
    machines: set,
    intervals: Dict[Tuple[int, int], Tuple[cp_model.IntervalVar, str]],
    machines_delays: Optional[Dict[str, Tuple[int, int]]] = None
) -> None:
    """
    Adds NoOverlap constraints for all machines, including optional fixed blocking intervals.

    :param model: The CP-SAT model instance.
    :param machines: Set of all machine names.
    :param intervals: Mapping from (job_idx, op_idx) to (IntervalVar, machine_name).
    :param machines_delays: Optional dictionary of blocking intervals per machine as (start, end).
    """
    for machine in machines:
        # Collect operation intervals for the machine
        machine_intervals = [
            interval for (_, _), (interval, machine_name) in intervals.items()
            if machine_name == machine
        ]

        # Add fixed blocking interval if defined for this machine
        if machines_delays and machine in machines_delays:
            start, end = machines_delays[machine]
            if end > start:
                fixed_interval = model.NewIntervalVar(start, end - start, end, f"fixed_{machine}")
                machine_intervals.append(fixed_interval)

        # Add NoOverlap constraint for the machine
        model.AddNoOverlap(machine_intervals)


def add_order_on_machines_deviation_terms(
        model: cp_model.CpModel, original_machine_orders: Dict[str, List[Tuple[str, int]]],
        operations: List[Tuple[int, str, int, int, str, int]],
        starts: Dict[Tuple[int, int], cp_model.IntVar]) -> List[Any]:
    """
    Add Boolean deviation terms for order violations on machines based on a previous schedule.

    For each machine, pairs of operations are compared according to their previous order.
    If the order is violated in the current schedule (i.e., the later-op starts before the earlier-op),
    a Boolean variable is set to 1 and added to the list of deviation terms.

    :param model: CpModel instance
    :param original_machine_orders: Dict mapping machine to ordered list of (job, op_id)
    :param operations: List of (job_idx, job, op_idx, op_id, machine, duration)
    :param starts: Dict mapping (job_idx, op_idx) to start time variables
    :return: List of BoolVar terms indicating order violations
    """
    deviation_terms = []

    # Build lookup from (job, op_id) to (job_idx, op_idx)
    job_op_to_index = {(job, op_id): (job_idx, op_idx)
                      for job_idx, job, op_idx, op_id, _, _ in operations}

    # For each machine sequence, add a violation var if the order is reversed
    for m, sequence in original_machine_orders.items():
        for i in range(len(sequence)):
            for j in range(i + 1, len(sequence)):
                job_a, op_id_a = sequence[i]
                job_b, op_id_b = sequence[j]

                if (job_a, op_id_a) in job_op_to_index and (job_b, op_id_b) in job_op_to_index:
                    a_idx = job_op_to_index[(job_a, op_id_a)]
                    b_idx = job_op_to_index[(job_b, op_id_b)]
                    a_start = starts[a_idx]
                    b_start = starts[b_idx]

                    violated = model.NewBoolVar(f"inv_{m}_{a_idx}_{b_idx}")
                    model.Add(b_start < a_start).OnlyEnforceIf(violated)
                    model.Add(b_start >= a_start).OnlyEnforceIf(violated.Not())
                    deviation_terms.append(violated)

    return deviation_terms

def add_kendall_tau_deviation_terms(
    model: cp_model.CpModel,
    original_machine_orders: Dict[str, List[Tuple[str, int]]],
    operations: List[Tuple[int, str, int, int, str, int]],
) -> List[Any]:

    deviation_terms = []
    pos_vars = {}

    # Mapping from (job, op_id) → (job_idx, op_idx)
    job_op_to_index = {
        (job, op_id): (job_idx, op_idx)
        for job_idx, job, op_idx, op_id, _, _ in operations
    }

    for machine, sequence in original_machine_orders.items():
        num_ops = len(sequence)

        # Erzeuge Positionsvariablen für alle Ops auf dieser Maschine
        machine_pos_vars = []
        for job, op_id in sequence:
            if (job, op_id) in job_op_to_index:
                job_idx, op_idx = job_op_to_index[(job, op_id)]
                pos_var = model.NewIntVar(0, num_ops - 1, f"pos_{machine}_{job_idx}_{op_idx}")
                pos_vars[(job_idx, op_idx)] = pos_var
                machine_pos_vars.append(pos_var)

        # Erzwinge eindeutige Permutation auf der Maschine
        if len(machine_pos_vars) >= 2:
            model.AddAllDifferent(machine_pos_vars)

        # Erzeuge Kendall-Inversionen
        for i in range(num_ops):
            for j in range(i + 1, num_ops):
                job_a, op_id_a = sequence[i]
                job_b, op_id_b = sequence[j]

                if (job_a, op_id_a) in job_op_to_index and (job_b, op_id_b) in job_op_to_index:
                    a_idx = job_op_to_index[(job_a, op_id_a)]
                    b_idx = job_op_to_index[(job_b, op_id_b)]

                    pos_a = pos_vars[a_idx]
                    pos_b = pos_vars[b_idx]

                    violated = model.NewBoolVar(f"kendall_inv_{machine}_{a_idx[0]}_{a_idx[1]}_vs_{b_idx[0]}_{b_idx[1]}")
                    model.Add(pos_a > pos_b).OnlyEnforceIf(violated)
                    model.Add(pos_a <= pos_b).OnlyEnforceIf(violated.Not())
                    deviation_terms.append(violated)

    return deviation_terms


def extract_cp_schedule_from_operations(
    operations: List[Tuple[int, str, int, int, str, int]],
    starts: Dict[Tuple[int, int], cp_model.IntVar],
    ends: Dict[Tuple[int, int], cp_model.IntVar],
    solver: cp_model.CpSolver
) -> List[Tuple[str, int, str, int, int, int]]:
    """
    Extracts the final schedule based on flattened operations and CP variables.

    :param operations: List of operations in the form (job_idx, job, op_idx, op_id, machine, duration).
    :type operations: List[Tuple[int, str, int, int, str, int]]
    :param starts: Dictionary of start time variables indexed by (job_idx, op_idx).
    :type starts: Dict[Tuple[int, int], cp_model.IntVar]
    :param ends: Dictionary of end time variables indexed by (job_idx, op_idx).
    :type ends: Dict[Tuple[int, int], cp_model.IntVar]
    :param solver: The CP-SAT solver instance after solving the model.
    :type solver: cp_model.CpSolver

    :return: List of (job, op_id, machine, start_time, duration, end_time) tuples.
    :rtype: List[Tuple[str, int, str, int, int, int]]
    """
    schedule = []
    for job_idx, job, op_idx, op_id, machine, duration in operations:
        start = solver.Value(starts[(job_idx, op_idx)])
        end = solver.Value(ends[(job_idx, op_idx)])
        schedule.append((job, op_id, machine, start, duration, end))
    return schedule

