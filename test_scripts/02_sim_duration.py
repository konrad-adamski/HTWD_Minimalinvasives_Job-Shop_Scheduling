import math
import time
from decimal import Decimal

from src.domain.Collection import LiveJobCollection
from src.domain.Query import JobQuery
from src.simulation.LognormalFactorGenerator import LognormalFactorGenerator

if __name__ == '__main__':
    max_util = 1.0
    shifts = 24
    source_name = "Fisher and Thompson 10x10"

    start = time.time()
    jobs = JobQuery.get_by_source_name_max_util_and_lt_arrival(
        source_name=source_name,
        max_bottleneck_utilization=Decimal(f"{max_util}"),
        arrival_limit=60 * 24 * shifts
    )
    end = time.time()
    print(f"Duration: {end - start}")

    jobs_collection = LiveJobCollection(jobs)


    jobs_collection.count_operations()
    print(jobs_collection.count_operations())

    start =time.time()
    factor_gen = LognormalFactorGenerator(sigma=0.2, seed=42)
    jobs_collection.sort_jobs_by_id()
    jobs_collection.sort_operations()

    for job in jobs_collection.values():
        for operation in job.operations:
            sim_duration_float = operation.duration * factor_gen.sample()
            operation.sim_duration = math.ceil(sim_duration_float)
    end = time.time()
    print(f"Duration: {end - start}")

