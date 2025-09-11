from decimal import Decimal

from src.domain.Collection import LiveJobCollection
from src.domain.Query import JobQuery, MachineInstanceQuery

if __name__ == '__main__':
    max_util = 0.85
    source_name = "Fisher and Thompson 10x10"

    jobs = JobQuery.get_by_source_name_and_max_bottleneck_utilization(
        source_name=source_name,
        max_bottleneck_utilization=Decimal(f"{max_util}")
    )

    for job in jobs[:2]:
        print(job)
        for operation in job.operations[:3]:
            print(operation)

    print("--" * 60)

    machines_instances = MachineInstanceQuery.get_by_source_name_and_max_bottleneck_utilization(
        source_name=source_name,
        max_bottleneck_utilization=Decimal(f"{max_util}")
    )

    for machine_instance in machines_instances:
        machine_name = machine_instance.name
        max_bottleneck_utilization = machine_instance.max_bottleneck_utilization
        transition_time = machine_instance.transition_time
        print(f"{machine_name=}, {max_bottleneck_utilization=}, {transition_time=}")

    print("--" * 60)

    jobs_collection = LiveJobCollection(jobs)

    for machine_instance in machines_instances:
        for job in jobs_collection.values():
            for operation in job.operations:
                if operation.machine_name == machine_instance.name:
                    operation.transition_time = machine_instance.transition_time


    for job in list(jobs_collection.values())[:2]:
        print(job)
        for operation in job.operations[:3]:
            transition_time = operation.transition_time
            print(f"\t{operation} {transition_time=}")