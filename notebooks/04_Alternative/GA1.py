# ================================================================
# GA für LiveJobCollection – Minimierung Makespan oder Sum Tardiness
# ================================================================
# pip install deap
from decimal import Decimal

from deap import base, creator, tools, algorithms
import random
from collections import defaultdict

from src.domain.Collection import LiveJobCollection
from src.domain.Query import JobQuery


# ---------- 1) Problem-Extraktion ----------
def _extract_problem_from_collection(jobs_collection):
    """
    Extrahiert Informationen aus LiveJobCollection.
    Gibt zurück:
      - jobs_ops: dict[job_id] -> [(machine_name, duration), ... in technologischer Reihenfolge]
      - earliest_start: dict[job_id] -> int
      - due_date: dict[job_id] -> int
      - machines: set[str]
    """
    jobs_ops = {}
    earliest_start = {}
    due_date = {}
    machines = set()
    for job in jobs_collection.values():
        earliest_start[job.id] = int(getattr(job, "earliest_start", 0) or 0)
        due_date[job.id] = int(getattr(job, "due_date", 0) or 0)
        ops_sorted = sorted(job.operations, key=lambda op: op.position_number)
        seq = []
        for op in ops_sorted:
            m = str(op.machine_name)
            d = int(op.duration)
            machines.add(m)
            seq.append((m, d))
        jobs_ops[job.id] = seq
    return jobs_ops, earliest_start, due_date, machines


# ---------- 2) Operationen eindeutig indizieren ----------
def _build_operation_index(jobs_ops):
    """
    Baut eine Liste aller Operationen.
    op_list: [(job_id, op_idx, machine, dur)]  mit op_id = Listenindex
    """
    op_list = []
    for j, ops in jobs_ops.items():
        for k, (m, d) in enumerate(ops):
            op_list.append((j, k, m, d))
    return op_list


# ---------- 3) Decoder mit Technologie + NoOverlap ----------
def decode_and_schedule_perm(perm, op_list, earliest_start):
    """
    Decodiert eine Permutation von op_ids in einen gültigen Zeitplan.
    - Permutation: jede Operation genau einmal (Integer 0..N-1)
    - Respektiert Reihenfolge innerhalb des Jobs (position_number)
    - Erzwingt NoOverlap über Maschinen- und Job-Freigabezeiten
    Rückgabe:
      makespan, schedule
      schedule = [(job_id, op_idx, machine, start, dur, end), ...]
    """
    next_needed = {j: 0 for j, *_ in op_list}
    machines = {m for (_, _, m, _) in op_list}

    machine_free = {m: 0 for m in machines}
    job_free = defaultdict(int)
    for j in next_needed:
        job_free[j] = earliest_start.get(j, 0)

    schedule = []
    queue = list(perm)
    while queue:
        op_id = queue.pop(0)
        j, k, m, d = op_list[op_id]
        if k != next_needed[j]:
            queue.append(op_id)  # Vorgänger noch nicht fertig → hinten anstellen
            continue
        start = max(machine_free[m], job_free[j])
        end = start + d
        machine_free[m] = end
        job_free[j] = end
        next_needed[j] += 1
        schedule.append((j, k, m, start, d, end))

    makespan = max(e for *_, e in schedule) if schedule else 0
    return makespan, schedule


# ---------- 4) Fitnessfunktion ----------
def _eval_individual(ind, op_list, earliest_start, due_date, objective="makespan"):
    mk, sched = decode_and_schedule_perm(ind, op_list, earliest_start)
    if objective == "makespan":
        return (mk,)
    # Sum Tardiness: Endzeit letzter Op eines Jobs vs. due_date
    job_end = defaultdict(int)
    for j, k, m, s, d, e in sched:
        job_end[j] = max(job_end[j], e)
    tard = sum(max(0, job_end[j] - due_date.get(j, 0)) for j in job_end)
    return (tard,)


# ---------- 5) GA-Wrapper ----------
def build_ga_for_live_collection(jobs_collection, objective="makespan",
                                 pop_size=80, ngen=120, cxpb=0.85, mutpb=0.2,
                                 seed=0, tournament=3):
    """
    Führt eine GA-Optimierung auf LiveJobCollection aus.
    objective: "makespan" oder "sum_tardiness"
    Rückgabe:
      best_value, best_schedule
      best_schedule = [(job_id, op_idx, machine, start, dur, end), ...] sortiert
    """
    random.seed(seed)
    jobs_ops, earliest_start, due_date, _ = _extract_problem_from_collection(jobs_collection)
    op_list = _build_operation_index(jobs_ops)
    n_ops = len(op_list)

    # Fitness/Individual nur einmalig anlegen
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Initialisierung
    def _init_individual():
        perm = list(range(n_ops))
        random.shuffle(perm)
        return creator.Individual(perm)

    def _evaluate(ind):
        return _eval_individual(ind, op_list, earliest_start, due_date, objective)

    toolbox.register("individual", _init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", _evaluate)
    toolbox.register("mate", tools.cxPartialyMatched)  # PMX funktioniert (Permutation von ints)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0 / max(1, n_ops))
    toolbox.register("select", tools.selTournament, tournsize=tournament)
    toolbox.register("map", map)  # robust für Multiprocessing-Kompatibilität

    # Evolution
    pop = toolbox.population(n=pop_size)
    algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)

    # Beste Lösung
    best = tools.selBest(pop, k=1)[0]
    best_val, best_sched = decode_and_schedule_perm(best, op_list, earliest_start)

    # Sortiert nach (machine, start)
    best_sched_sorted = sorted(best_sched, key=lambda r: (r[2], r[3]))
    return best_val, best_sched_sorted


# ---------- 6) Beispielaufruf ----------
if __name__ == "__main__":

    jobs = JobQuery.get_by_source_name_max_util_and_lt_arrival(
        source_name="Fisher and Thompson 10x10",
        max_bottleneck_utilization=Decimal("0.90"),
        arrival_limit=60 * 24 * 2  # 2 days
    )
    # jobs = [job for job in all_jobs if job.earliest_start <=timespan]

    jobs_collection = LiveJobCollection(jobs)

    best_val, best_sched = build_ga_for_live_collection(
        jobs_collection,
        objective="sum_tardiness",  # oder "sum_tardiness"
        pop_size=100,  # Größe der Population
        ngen=200,  # Anzahl Generationen
        cxpb=0.8,  # Crossover-Wahrscheinlichkeit
        mutpb=0.2,  # Mutations-Wahrscheinlichkeit
        seed=42,  # Zufalls-Seed für Reproduzierbarkeit
        tournament=3  # Turniergröße
    )

    print("Beste Zielfunktion:", best_val)
    print("Beste Lösung (job_id, op_idx, machine, start, duration, end):")
    for rec in best_sched:
        print(rec)
