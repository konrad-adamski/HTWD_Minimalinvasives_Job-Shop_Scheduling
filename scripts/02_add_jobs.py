from typing import List

from src.classes.Query import RoutingQuery
from src.classes.Initializer import JobsInitializer
from src.classes.orm_models import Job
from src.classes.orm_setup import SessionLocal

if __name__ == "__main__":

    routings = RoutingQuery.get_by_source_name(source_name="Fisher and Thompson 10x10")

    max_bottleneck_utilization_list = [0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

    all_jobs: List[Job] = []

    for max_utilization in max_bottleneck_utilization_list:
        mean_arrival_time = JobsInitializer._calculate_mean_interarrival_time(routings, u_b_mmax=max_utilization)
        print(f"\nMean interarrival time for {max_utilization}: {mean_arrival_time}")

        jobs = JobsInitializer.create_jobs(
            routings=routings,
            max_bottleneck_utilization=max_utilization,
            total_shift_number=500
        )
        all_jobs.extend(jobs)
        
    for job in all_jobs[:3]:
        print(job)
        for operation in job.operations:
            print(f" {operation}")

    with SessionLocal() as session:
        session.add_all(all_jobs)
        session.commit()

