from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd

from omega.db_models import Experiment, Routing, Job, RoutingSource


class ExperimentBuilder:
    def create_experiment(
            self, solver_main_pct: float, solver_w_t:int, solver_w_e:int,
            solver_w_first:int, max_bottleneck_utilization: float,
            sim_sigma: float, routings: List[Routing]):
        experiment = Experiment(
            main_pct=solver_main_pct,
            w_t=solver_w_t,
            w_e=solver_w_e,
            w_first=solver_w_first,
            max_bottleneck_utilization=max_bottleneck_utilization,
            sim_sigma=sim_sigma
        )

        jobs: List[Job] = []
        for routing in routings:
            Job(id=f"J{routing.id.zfill(2)}{experiment.id:2}",routing=routing, arrival=0,
                earliest_start=1440,
                deadline=2800,
                experiment=experiment
            )

        return experiment

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


    def _get_vec_t_b_mmax_from_routings(
            self, routings: List[Routing],
            verbose: bool = False
    ) -> List[int]:
        """
        Return the processing time of each routing on the bottleneck machine.

        The list is ordered according to the order of the `routings` list.

        :param routings: List of Routing objects, each with a .operations list
        :param verbose: If True, prints total usage per machine
        :return: List of processing times on the bottleneck machine (0 if not used)
        """
        # 1) determine the bottleneck machine
        bottleneck_machine = self._get_bottleneck_machine_from_routings(routings, verbose=verbose)

        # 2) for each routing, sum durations on that machine
        vec = []
        for routing in routings:
            t_b = sum(op.duration for op in routing.operations if op.machine == bottleneck_machine)
            vec.append(t_b)
        return vec

    def calculate_mean_interarrival_time(
        self, routings: List[Routing], u_b_mmax: float = 0.9, p: Optional[List[float]] = None,
        verbose: bool = False) -> float:
        """
        Berechnet die mittlere Zwischenankunftszeit t_a, sodass die gewünschte Auslastung der Engpassmaschine erreicht wird.

        :param routings: Liste von Routing-Objekten
        :param u_b_mmax: Zielauslastung der Engpassmaschine (z. B. 0.9)
        :param verbose: Wenn True, werden Details ausgegeben
        :return: t_a, gerundet auf 2 Nachkommastellen
        """
        n = len(routings)
        if n == 0:
            raise ValueError("Routingliste darf nicht leer sein.")


        # 2) Bearbeitungszeiten auf Engpassmaschine
        vec_t_b_mmax = self._get_vec_t_b_mmax_from_routings(routings, verbose=verbose)

        if verbose:
            print(f"Bearbeitungszeiten auf der Engpassmaschine: {vec_t_b_mmax}")

        # 3) Erwartete Bearbeitungszeit / Zielauslastung
        if p is not None and len(p) == len(vec_t_b_mmax):
            t_a = sum(p[i] * vec_t_b_mmax[i] for i in range(n)) / u_b_mmax
        else:
            # Gleichverteilte Routing - Wahrscheinlichkeit p = [1.0 / n] * n
            t_a = np.mean(vec_t_b_mmax) / u_b_mmax
        return round(t_a, 2)


if __name__ == "__main__":
    # RoutingSource erzeugen
    routing_source = RoutingSource(name="Testdatensatz")

    # Example with multiple Routings
    data = {
        "Routing_ID": ["R1", "R1", "R2", "R2"],
        "Operation": [10, 20, 10, 20],
        "Machine": ["M1", "M2", "M3", "M1"],
        "Processing Time": [5, 10, 7, 14]
    }
    dframe_routings = pd.DataFrame(data)

    # Routings aus DataFrame erzeugen
    routings = Routing.from_multiple_routings_dataframe(dframe_routings, source=routing_source)


    builder = ExperimentBuilder()

    engpass_machine = builder._get_engpassmaschine_from_routings(routings, verbose=True)
    print(f"Engpassmachine: {engpass_machine}")