import random
import os
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

from deap import base, creator, tools, algorithms

from src.Logger import Logger
from src.domain.Collection import LiveJobCollection


class Solver:
    def __init__(self, jobs_collection: LiveJobCollection, logger: Logger, schedule_start: int = 0):
        self.logger = logger
        self.jobs_collection = jobs_collection
        self.schedule_start = schedule_start

        # Ergebnisvariablen
        self.best_value: Optional[int] = None
        self.best_individual = None
        self.best_schedule: List[Tuple[str, int, str, int, int, int]] = []
        self.model_completed = False

    # -----------------------------------------------------------
    # Hilfsfunktionen
    # -----------------------------------------------------------
    def _extract_problem(self):
        jobs_ops = {}
        earliest_start = {}
        due_date = {}
        machines = set()
        for job in self.jobs_collection.values():
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

    def _build_operation_index(self, jobs_ops):
        op_list = []
        for j, ops in jobs_ops.items():
            for k, (m, d) in enumerate(ops):
                op_list.append((j, k, m, d))
        return op_list

    def _decode(self, perm, op_list, earliest_start):
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
                queue.append(op_id)
                continue
            start = max(machine_free[m], job_free[j])
            end = start + d
            machine_free[m] = end
            job_free[j] = end
            next_needed[j] += 1
            schedule.append((j, k, m, start, d, end))

        makespan = max(e for *_, e in schedule) if schedule else 0
        return makespan, schedule

    def _evaluate(self, ind, op_list, earliest_start, due_date, objective="makespan"):
        mk, sched = self._decode(ind, op_list, earliest_start)
        if objective == "makespan":
            return (mk,)
        job_end = defaultdict(int)
        for j, k, m, s, d, e in sched:
            job_end[j] = max(job_end[j], e)
        tard = sum(max(0, job_end[j] - due_date.get(j, 0)) for j in job_end)
        return (tard,)

    # -----------------------------------------------------------
    # GA Solver
    # -----------------------------------------------------------
    def solve_model(self,
                    objective: str = "makespan",
                    pop_size: int = 80,
                    ngen: int = 120,
                    cxpb: float = 0.85,
                    mutpb: float = 0.2,
                    seed: int = 0,
                    tournament: int = 3):

        random.seed(seed)
        jobs_ops, earliest_start, due_date, _ = self._extract_problem()
        op_list = self._build_operation_index(jobs_ops)
        n_ops = len(op_list)

        if "FitnessMin" not in creator.__dict__:
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        def _init_individual():
            perm = list(range(n_ops))
            random.shuffle(perm)
            return creator.Individual(perm)

        def _eval(ind):
            return self._evaluate(ind, op_list, earliest_start, due_date, objective)

        toolbox.register("individual", _init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", _eval)
        toolbox.register("mate", tools.cxPartialyMatched)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0 / max(1, n_ops))
        toolbox.register("select", tools.selTournament, tournsize=tournament)
        toolbox.register("map", map)

        pop = toolbox.population(n=pop_size)
        algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)

        best = tools.selBest(pop, k=1)[0]
        self.best_value, self.best_schedule = self._decode(best, op_list, earliest_start)
        self.best_individual = best
        self.model_completed = True

    # -----------------------------------------------------------
    # Ergebnisse
    # -----------------------------------------------------------
    def get_schedule(self) -> Optional[LiveJobCollection]:
        if not self.model_completed:
            self.logger.warning("Model not solved yet.")
            return None

        sched_collection = LiveJobCollection()
        for j, k, m, s, d, e in self.best_schedule:
            # Hole originale Operation aus jobs_collection
            job = self.jobs_collection[j]
            operation = job.operations[k]
            sched_collection.add_operation_instance(op=operation, new_start=s, new_end=e)
        return sched_collection

    def get_solver_info(self) -> dict:
        if not self.model_completed:
            return {"access_fault": "Model is not complete!"}
        return {"objective_value": self.best_value}
