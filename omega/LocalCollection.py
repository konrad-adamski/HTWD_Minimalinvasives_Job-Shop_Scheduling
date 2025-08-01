from __future__ import annotations

import pandas as pd
from collections import UserDict
from typing import List, Optional
from omega.db_models import Routing, RoutingSource, Job, Experiment


class RoutingCollection(UserDict):
    def __init__(self, routings: List[Routing]):
        # Dict mit ID als Schlüssel
        super().__init__({routing.id: routing for routing in routings})

    def get_routing(self, routing_id: str) -> Optional[Routing]:
        """Finde Routing anhand der ID."""
        return self.data.get(routing_id)

    def all_ids(self) -> List[str]:
        """Gibt alle Routing-IDs zurück."""
        return list(self.data.keys())

    def total_operations(self) -> int:
        """Gesamtzahl aller enthaltenen Operationen."""
        return sum(len(r.operations) for r in self.data.values())

class JobCollection(UserDict):
    def __init__(self, jobs: Optional[List[Job]] = None):
        super().__init__()
        if jobs:
            for job in jobs:
                self.add(job)

    def add(self, job: Job):
        if not isinstance(job, Job):
            raise TypeError(f"Expected Job, got {type(job)}")
        self.data[job.id] = job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.data.get(job_id)

    def get_jobs_by_earliest_start(self, earliest_start: int) -> List[Job]:
        return [job for job in self.values() if job.earliest_start == earliest_start]


    def get_subset(self, earliest_start: int, planable_job_ids: Optional[List[str]] = None)-> JobCollection:
        """
        Returns a subset with jobs that are either newly arrived or have unscheduled operations.

        :param earliest_start: Current time (selects newly arrived jobs)
        :param planable_job_ids: Jobs with pending operations
        :return: Filtered JobInformationCollection
        """
        subset = JobCollection()
        for job_id, info in self.items():
            if info.earliest_start == earliest_start or (planable_job_ids and job_id in planable_job_ids):
                subset[job_id] = info
        return subset



    def __repr__(self):
        return f"<JobCollection ({len(self)} jobs)>"


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

    routing_collection = RoutingCollection(routings)

    r1 = routing_collection.get_routing("R1")
    print(f"R1 hat {len(r1.operations)} Operations.")

    print(f"Alle Routing-IDs: {routing_collection.all_ids()}")
    print(f"Total Operationen: {routing_collection.total_operations()}")


    for routing in routing_collection.values():
        print(f"Routing-ID: {routing.id} from {routing.source_name} ({routing.source_id})")
        print(f"Gesamtdauer: {routing.sum_duration} min")

        for op in routing.operations:
            print(f"  • Step {op.position_number}: {op.machine}, {op.duration} min")

    print("-"*80)
    # Job Collection --------------------------------------------------------------------------------------------------

    experiment1 = Experiment()
    job1 = Job(id="J1", routing=r1, arrival=0, earliest_start=1440,deadline=2800, experiment=experiment1)
    job11 = Job(id="J11", routing=r1, arrival=0, earliest_start=1840, deadline=3800, experiment=experiment1)

    jobs_collection = JobCollection([job1, job11])

    # 5. Ausgabe
    for job in jobs_collection.values():
        print(f"\nJob {job.id} (Routing: {job.routing_id}) mit {len(job.operations)} Operationen:")
        for op in job.operations:
            print(f"  – Step {op.position_number}: {op.machine}, {op.duration} min. "
                  f"Job earliest_start: {op.job_earliest_start}")

