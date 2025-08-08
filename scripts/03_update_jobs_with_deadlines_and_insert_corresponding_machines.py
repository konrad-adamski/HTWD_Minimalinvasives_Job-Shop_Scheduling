from decimal import Decimal

from src.DataFrameEnrichment import DataFrameEnrichment as DataEnrichment
from src.domain.Collection import LiveJobCollection
from src.domain.Initializer import MachineInitializer
from src.domain.Query import JobQuery
from src.simulation.ProductionSimulation import ProductionSimulation

if __name__ == "__main__":
    source_name = "Fisher and Thompson 10x10"
    max_bottleneck_utilization_list = [0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

    for max_bottleneck_utilization in max_bottleneck_utilization_list:

        jobs = JobQuery.get_by_source_name_and_max_bottleneck_utilization(
            source_name=source_name,
            max_bottleneck_utilization=Decimal(f"{max_bottleneck_utilization}")
        )

        # Collection with jobs
        jobs_collection = LiveJobCollection(jobs)

        # Simulation
        simulation = ProductionSimulation(sigma=0, verbose=False, with_earliest_start=True)
        simulation.run(jobs_collection, start_time=0, end_time=None)

        finished_operations = simulation.get_finished_operation_collection()
        df_fifo_schedule = finished_operations.to_operations_dataframe()

        df_jobs_times_temp = finished_operations.to_jobs_metrics_dataframe()

        # Generation of deadlines using log-normal distribution -------------------------------------------------------
        df_jobs_times = DataEnrichment.add_groupwise_lognormal_deadlines_by_group_mean(
            df_times_temp = df_jobs_times_temp,
            sigma= 0.25
        )

        df_jobs_times_final = DataEnrichment.ensure_reasonable_deadlines(df_jobs_times, min_coverage=1.0)
        JobQuery.update_job_deadlines_from_df(
            df=df_jobs_times_final,
            job_column="Job",
            deadline_column="Deadline"
        )

        # Transition Times --------------------------------------------------------------------------------------------
        df_avg_transition_times = DataEnrichment.compute_avg_transition_times_per_machine_backward(df_fifo_schedule)

        MachineInitializer.insert_from_dataframe(
            df=df_avg_transition_times,
            source_name=source_name,
            max_bottleneck_utilization=Decimal(f"{max_bottleneck_utilization}")
        )






