from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import inspect
from sqlalchemy.orm import joinedload, InstrumentedAttribute

from omega.db_models import Experiment, Routing, Job, RoutingSource
from omega.db_setup import SessionLocal, my_engine


class ExperimentBuilder:
    def __new__(cls, *args, **kwargs):
        raise TypeError("ExperimentBuilder is a static utility class and cannot be instantiated.")

    @staticmethod
    def _get_bottleneck_machine_from_routings(
            routings: List[Routing],
            verbose: bool = False
    ) -> str:
        """
        Identifiziert die Engpassmaschine (höchste Gesamtauslastung) aus einer Liste von Routing-Objekten.

        :param routings: Liste von Routing-Objekten, die jeweils eine Liste von RoutingOperations enthalten
        :param verbose: Bei True werden die Maschinenlasten ausgegeben
        :return: Maschinenname mit der höchsten Gesamtauslastung
        """
        usage = defaultdict(int)

        for routing in routings:
            for op in routing.operations:
                usage[op.machine] += op.duration

        if verbose:
            print("Maschinenlast (Gesamte Bearbeitungszeit):")
            for machine, total in sorted(usage.items(), key=lambda x: str(x[0])):
                print(f"  {machine}: {total}")

        if not usage:
            raise ValueError("Keine Maschinenbelastung gefunden – Liste war eventuell leer oder ungültig.")

        return max(usage, key=usage.get)

    @classmethod
    def _get_vec_t_b_mmax_from_routings(cls, routings: List[Routing], verbose: bool = False) -> List[int]:
        """
        Return the processing time of each routing on the bottleneck machine.

        The list is ordered according to the order of the `routings` list.

        :param routings: List of Routing objects, each with a .operations list
        :param verbose: If True, prints total usage per machine
        :return: List of processing times on the bottleneck machine (0 if not used)
        """
        # 1) determine the bottleneck machine
        bottleneck_machine = cls._get_bottleneck_machine_from_routings(routings, verbose=verbose)

        # 2) for each routing, sum durations on that machine
        vec = []
        for routing in routings:
            t_b = sum(op.duration for op in routing.operations if op.machine == bottleneck_machine)
            vec.append(t_b)
        return vec

    @classmethod
    def calculate_mean_interarrival_time(
        cls, routings: List[Routing], u_b_mmax: float = 0.9, p: Optional[List[float]] = None,
        verbose: bool = False) -> float:
        """
        Berechnet die mittlere Zwischenankunftszeit t_a, sodass die gewünschte Auslastung der Engpassmaschine erreicht wird.

        :param routings: Liste von Routing-Objekten
        :param u_b_mmax: Zielauslastung der Engpassmaschine (z.B. 0.9)
        :param verbose: Wenn True, werden Details ausgegeben
        :return: t_a, gerundet auf 2 Nachkommastellen
        """
        n = len(routings)
        if n == 0:
            raise ValueError("Routingliste darf nicht leer sein.")


        # 2) Bearbeitungszeiten auf Engpassmaschine
        vec_t_b_mmax = cls._get_vec_t_b_mmax_from_routings(routings, verbose=verbose)

        if verbose:
            print(f"Bearbeitungszeiten auf der Engpassmaschine: {vec_t_b_mmax}")

        # 3) Erwartete Bearbeitungszeit / Zielauslastung
        if p is not None and len(p) == len(vec_t_b_mmax):
            t_a = sum(p[i] * vec_t_b_mmax[i] for i in range(n)) / float(u_b_mmax)
        else:
            # Gleichverteilte Routing - Wahrscheinlichkeit p = [1.0 / n] * n
            t_a = np.mean(vec_t_b_mmax) / float(u_b_mmax)
        return round(t_a, 4)

    @staticmethod
    def gen_arrivals(
            mean_interarrival_time: float, size: int, start_time: int = 0,
            last_arrival_time: int = 1440, random_seed: Optional[int] = 120)-> List[int]:

        if random_seed is not None:
            np.random.seed(random_seed)

        # 1) Generate exponentially distributed interarrival times
        interarrival_times = np.random.exponential(scale=mean_interarrival_time, size=size)
        interarrival_times[0] = start_time

        # 2) Calculate cumulative arrival times
        arrivals = np.cumsum(interarrival_times)
        arrivals = np.floor(arrivals).astype(int)

        # 3) Remove values greater than last_arrival_time
        arrivals = arrivals[arrivals <= last_arrival_time]

        return np.floor(arrivals).astype(int).tolist()


    @classmethod
    def create_jobs(
            cls, routings: List[Routing], experiment: Experiment,
            shift_count: int = 1, arrival_seed: Optional[int] = 120,
            job_routing_seed: Optional[int] = 100, verbose: bool = False):

        jobs: List[Job] = []
        last_shift_end = 1440 * (shift_count+1)

        # Compute mean interarrival time based on max bottleneck utilization
        mean_arrival_time = cls.calculate_mean_interarrival_time(
            routings=routings,
            u_b_mmax=experiment.max_bottleneck_utilization,
            verbose=verbose
        )

        approx_size = np.ceil(last_shift_end / mean_arrival_time).astype(int)

        arrivals = cls.gen_arrivals(
            mean_interarrival_time=mean_arrival_time,
            size=approx_size,
            last_arrival_time=last_shift_end,
            random_seed=arrival_seed
        )

        a_idx = 0
        while a_idx < len(arrivals):
            temp_routings = routings.copy()

            if job_routing_seed is not None:
                np.random.seed(job_routing_seed+a_idx)
                np.random.shuffle(temp_routings)

            for i, routing in enumerate(temp_routings):
                if a_idx >= len(arrivals):
                    break

                job = Job(id=f"J{experiment.id:03d}-{a_idx:04d}",
                    routing=routing,
                    arrival=arrivals[a_idx],
                    deadline=None,
                    experiment=experiment
                )
                jobs.append(job)
                a_idx += 1
        return jobs



    @staticmethod
    def add_experiment(solver_main_pct: float = 0.5, solver_w_t:int = 10, solver_w_e:int = 2,
            solver_w_first:int = 1, max_bottleneck_utilization: float = 0.90, sim_sigma: float  = 0.25) -> Experiment:
        with SessionLocal() as session:
            experiment = Experiment(
                main_pct=solver_main_pct,
                w_t=solver_w_t,
                w_e=solver_w_e,
                w_first=solver_w_first,
                max_bottleneck_utilization=max_bottleneck_utilization,
                sim_sigma=sim_sigma
            )
            session.add(experiment)
            session.commit()

            return session.get(Experiment, experiment.id)

    @classmethod
    def insert_jobs(cls, routings: List[Routing], experiment: Experiment,
            shift_count: int = 1, arrival_seed: Optional[int] = 120,
            job_routing_seed: Optional[int] = 100, verbose: bool = False):

        jobs = cls.create_jobs(
            routings=routings,
            experiment=experiment,
            shift_count=shift_count,
            arrival_seed=arrival_seed,
            job_routing_seed=job_routing_seed,
            verbose=verbose
        )

        with SessionLocal() as session:
            session.add_all(jobs)
            session.commit()



if __name__ == "__main__":
    from configs.path_manager import get_path

    # RoutingSource erzeugen
    routing_source = RoutingSource(name="FT 10x10")


    basic_data_path = get_path("data", "basic")
    df_routings = pd.read_csv(basic_data_path / "ft10_routings.csv")


    # Routings aus DataFrame erzeugen
    routings = Routing.from_multiple_routings_dataframe(df_routings, source=routing_source)

    builder = ExperimentBuilder()

    mean_arrival_time = builder.calculate_mean_interarrival_time(routings, u_b_mmax=0.9, verbose=True)
    print(f"\nMean interarrival time: {mean_arrival_time}")

    print("-"*80)

    experiment = builder.add_experiment_to_db(max_bottleneck_utilization=0.9)
    print(f"\nExperiment ID: {experiment.id}")



    jobs = builder.create_jobs(
        routings=routings,
        experiment=experiment
    )

    jobsis = builder.add_jobs_to_db(jobs)

    for job in jobsis[:11]:
        print(f"Job {job.id}, {job.routing_id}")
        #for op in job.operations:
    #    print(f"\t|{op.position_number}\t|{op.machine}\t|duration {op.duration}")
        
