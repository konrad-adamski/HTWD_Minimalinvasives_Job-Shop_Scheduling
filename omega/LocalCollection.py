from collections import UserDict
from typing import List, Optional

import pandas as pd

from omega.db_models import Routing, RoutingSource, Job


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

    def get_by_routing(self, routing_id: str) -> List[Job]:
        return [job for job in self.values() if job.routing_id == routing_id]

    def filter_by_deadline(self, max_deadline: int) -> List[Job]:
        return [job for job in self.values() if job.deadline <= max_deadline]

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
    job1 = Job(id="J1", routing=r1, arrival=0, earliest_start=1440,deadline=2800)
    job11 = Job(id="J11", routing=r1, arrival=0, earliest_start=1840, deadline=3800)

    jobs = JobCollection([job1, job11])

    # 5. Ausgabe
    for job in jobs.values():
        print(f"\nJob {job.id} (Routing: {job.routing_id}) mit {len(job.operations)} Operationen:")
        for op in job.operations:
            print(f"  – Step {op.position_number}: {op.machine}, {op.duration} min. "
                  f"Job earliest_start: {op.job_earliest_start}")

