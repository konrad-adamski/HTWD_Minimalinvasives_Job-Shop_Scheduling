import time
import random
from src.domain.Collection import LiveJobCollection


def solve(
    jobs_collection: LiveJobCollection,
    priority_rule: str = "SPT",   # "SPT", "LPT", "FCFS", "RANDOM", "EDD"
    log_on: bool = True
):
    """
    Giffler-Thompson für LiveJobCollection.

    Erwartete Struktur:
      for job in jobs_collection.values():
          job.id, job.earliest_start, job.due_date
          for op in job.operations:
              op.position_number, op.machine_name, op.duration

    Rückgabe:
      schedule_job_collection: LiveJobCollection mit gesetzten Start/Ende via add_operation_instance(...)
    """
    starting_time = time.time()
    if log_on:
        print(f'Giffler-Thompson-Algorithmus mit Prioritätsregel "{priority_rule}" gestartet ...\n')

    # --- 1) Daten extrahieren
    jobs = list(jobs_collection.values())
    job_due = {job.id: int(job.due_date) for job in jobs}
    job_ops = {}            # job_id -> [(op_id, machine, dur, op_obj)]
    machines = set()

    for job in jobs:
        ops_sorted = sorted(job.operations, key=lambda o: o.position_number)
        job_ops[job.id] = [(op.position_number, op.machine_name, int(op.duration), op) for op in ops_sorted]
        for op in ops_sorted:
            machines.add(op.machine_name)

    # --- 2) Zeitstatus
    job_op_ready = {job.id: int(job.earliest_start) for job in jobs}
    machine_available = {m: 0 for m in machines}
    job_op_index = {job.id: 0 for job in jobs}

    # FCFS-Tie-Break: earliest_start, dann job.id
    fcfs_order = sorted([(job.id, job_op_ready[job.id]) for job in jobs], key=lambda t: (t[1], str(t[0])))

    # Ergebnis-Collection
    schedule_job_collection = LiveJobCollection()
    remaining = sum(len(v) for v in job_ops.values())

    # --- 3) Iteratives Einplanen
    while remaining > 0:
        candidates = []
        for job in jobs:
            j = job.id
            idx = job_op_index[j]
            if idx < len(job_ops[j]):
                op_id, m, d, op_obj = job_ops[j][idx]
                est = max(job_op_ready[j], machine_available[m])
                candidates.append((est, d, j, op_id, m, op_obj))

        min_est = min(c[0] for c in candidates)
        conflict_ops = [c for c in candidates if c[0] == min_est]

        # --- 4) Auswahlregel
        if priority_rule == "SPT":
            selected = min(conflict_ops, key=lambda x: x[1])
        elif priority_rule == "LPT":
            selected = max(conflict_ops, key=lambda x: x[1])
        elif priority_rule == "FCFS":
            fcfs_rank = {jid: i for i, (jid, _) in enumerate(fcfs_order)}
            selected = min(conflict_ops, key=lambda x: (fcfs_rank[x[2]], x[1], str(x[2])))
        elif priority_rule == "EDD":
            selected = min(conflict_ops, key=lambda x: (job_due[x[2]], x[1], str(x[2])))
        else:  # "RANDOM"
            selected = random.choice(conflict_ops)

        est, d, j, op_id, m, op_obj = selected
        start = est
        end = start + d

        # --- 5) Direkt ins Ergebnis schreiben
        schedule_job_collection.add_operation_instance(
            op=op_obj,
            new_start=start,
            new_end=end
        )

        # Status fortschreiben
        job_op_ready[j] = end
        machine_available[m] = end
        job_op_index[j] += 1
        remaining -= 1

    # --- 6) Logging
    if log_on:
        print("\nPlan-Informationen:")
        print(f"  Anzahl Jobs        : {len(jobs)}")
        print(f"  Anzahl Maschinen   : {len(machines)}")
        print(f"  Laufzeit           : ~{time.time() - starting_time:.2f} Sekunden")

    return schedule_job_collection