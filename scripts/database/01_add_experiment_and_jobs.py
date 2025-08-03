from omega.Builder import ExperimentBuilder
from omega.QueryHelper import RoutingQuery, JobQuery

if __name__ == "__main__":

    routings = RoutingQuery.get_by_source_name(source_name="Fisher and Thompson 10x10")

    for routing in routings[:2]:
        print(routing)
        for operation in routing.operations:
            print(operation)

    print("-"*80, end="\n\n")
    mean_arrival_time = ExperimentBuilder.calculate_mean_interarrival_time(routings, u_b_mmax=0.85, verbose=True)
    print(f"\nMean interarrival time: {mean_arrival_time}")


    experiment = ExperimentBuilder.add_experiment(max_bottleneck_utilization=0.9)
    experiment_id = experiment.id
    print(f"\nExperiment ID: {experiment.id}, {experiment.max_bottleneck_utilization}")

    print("-" * 80, end="\n\n")
    ExperimentBuilder.insert_jobs(routings=routings,experiment=experiment)


    jobs = JobQuery.get_by_experiment_id(experiment_id=experiment_id)
    for job in jobs[:2]:
        print(f"{job}")
        for op in job.operations:
            print(f"\t {op}")