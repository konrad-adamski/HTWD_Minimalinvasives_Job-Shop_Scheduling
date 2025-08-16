from src.domain.Query import RoutingQuery
from src.domain.Initializer import JobsInitializer


if __name__ == "__main__":

    max_bottleneck_utilization_list = [0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

    for max_utilization in max_bottleneck_utilization_list:
        routings = RoutingQuery.get_by_source_name(source_name="Fisher and Thompson 10x10")

        # noinspection PyProtectedMember
        mean_arrival_time = JobsInitializer._calculate_mean_interarrival_time(routings, u_b_mmax=max_utilization)
        print(f"\n--- Mean inter-arrival time: {mean_arrival_time} ---")


        JobsInitializer.insert_jobs(
            routings=routings,
            max_bottleneck_utilization=max_utilization,
            total_shift_number=400
        )

