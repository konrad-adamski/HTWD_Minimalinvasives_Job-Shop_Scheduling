from src.classes.Collections import RoutingsCollection, JobsCollection
from src.classes.Initializer import ExperimentInitializer

if __name__ == "__main__":

    routings_collection = RoutingsCollection.from_db_by_source_name(source_name="Fisher and Thompson 10x10")

    for routing in list(routings_collection.values())[:2]:
        print(f"\n{routing}")
        for operation in routing.operations:
            print(f"\t{operation}")

    print("-"*80, end="\n\n")

    routings = routings_collection.get_routings()

    mean_arrival_time = ExperimentInitializer._calculate_mean_interarrival_time(routings, u_b_mmax=0.85, verbose=True)
    print(f"\nMean interarrival time: {mean_arrival_time}")


    experiment = ExperimentInitializer.add_experiment(max_bottleneck_utilization=0.9)
    experiment_id = experiment.id
    print(f"\nExperiment ID: {experiment.id}, {experiment.max_bottleneck_utilization}")

    print("-" * 80, end="\n\n")
    ExperimentInitializer.insert_jobs(routings=routings,experiment=experiment)

    jobs_collection = JobsCollection.from_db_by_experiment_id(experiment_id=experiment_id)

    jobs = list(jobs_collection.values())
    for job in jobs[:2]:
        print(f"{job}")
        for op in job.operations:
            print(f"\t {op}")
