# pip install deap
from deap import base, creator, tools, algorithms
import random

# --- Problem: kleines JSSP-Beispiel ---
# jobs[j] = Liste von Operationen (machine, duration) in Reihenfolge
jobs = {
    0: [(0, 3), (1, 2), (2, 2)],
    1: [(0, 2), (2, 1), (1, 4)],
    2: [(1, 4), (2, 3), (0, 1)],
}
num_jobs = len(jobs)
num_ops_total = sum(len(ops) for ops in jobs.values())
machines = {m for ops in jobs.values() for (m, _) in ops}

# --- GA-Genotyp: Permutation von Job-IDs in Länge num_ops_total
# Interpretation: wann immer Job j erscheint, plane die nächste Operation von j.
def decode_and_schedule(sequence):
    # Fortschritt je Job (welche op-id als nächstes?)
    next_op = {j:0 for j in jobs}
    # Maschinen-/Job-Verfügbarkeit (Endzeit der letzten belegten Operation)
    machine_free = {m:0 for m in machines}
    job_free = {j:0 for j in jobs}
    schedule = []  # (job, op_id, machine, start, dur, end)

    for j in sequence:
        op_id = next_op[j]
        if op_id >= len(jobs[j]):
            # Falls Job schon fertig ist, ignoriere (robustheitshalber)
            continue
        m, d = jobs[j][op_id]
        start = max(machine_free[m], job_free[j])  # NoOverlap by construction
        end = start + d
        # Update Zustände
        machine_free[m] = end
        job_free[j] = end
        next_op[j] += 1
        schedule.append((j, op_id, m, start, d, end))

    # Wenn nicht alle Operationen eingeplant wurden (z.B. zu viele Wiederholungen),
    # plane Rest sequenziell:
    for j in jobs:
        while next_op[j] < len(jobs[j]):
            m, d = jobs[j][next_op[j]]
            start = max(machine_free[m], job_free[j])
            end = start + d
            machine_free[m] = end
            job_free[j] = end
            schedule.append((j, next_op[j], m, start, d, end))
            next_op[j] += 1

    makespan = max(e for *_, e in schedule)
    return makespan, schedule

# --- DEAP Setup ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Individuum: Multiset-Permutation = jedes Joblabel kommt so oft vor wie #Ops
job_multiset = []
for j, ops in jobs.items():
    job_multiset += [j]*len(ops)

def init_ind():
    x = job_multiset[:]
    random.shuffle(x)
    return creator.Individual(x)

toolbox.register("individual", init_ind)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_ind(ind):
    makespan, _ = decode_and_schedule(ind)
    return (makespan,)

toolbox.register("evaluate", eval_ind)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- Lauf ---
random.seed(0)
pop = toolbox.population(n=60)
algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.2, ngen=80, verbose=False)
best = tools.selBest(pop, k=1)[0]
best_mk, best_sched = decode_and_schedule(best)

print("Bestes Makespan:", best_mk)
for rec in sorted(best_sched, key=lambda r: (r[2], r[3])):  # sortiert nach Maschine/Start
    print(rec)
