from decimal import Decimal

from src.DataFrameEnrichment import DataFrameEnrichment as DataEnrichment, DataFrameEnrichment
from src.domain.Collection import LiveJobCollection
from src.domain.Initializer import MachineInstanceInitializer
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

        # Add simulation durations to operations
        for job in jobs_collection.values():
            for operation in job.operations:
                operation.sim_duration = operation.duration  # sigma = 0

        # Simulation
        simulation = ProductionSimulation(verbose=False, with_earliest_start=True)
        simulation.run(jobs_collection, start_time=0, end_time=None)

        finished_operations = simulation.get_finished_operation_collection()
        df_fifo_schedule = finished_operations.to_operations_dataframe()

        df_jobs_times_temp = finished_operations.to_jobs_metrics_dataframe()

        # Generation of due_dates using log-normal distribution -------------------------------------------------------
        df_jobs_times = DataEnrichment.add_groupwise_lognormal_due_dates_by_group_mean(
            df_times_temp = df_jobs_times_temp,
            sigma= 0.25
        )

        df_jobs_times_final = DataEnrichment.ensure_reasonable_due_dates(df_jobs_times, min_coverage=1.0)
        JobQuery.update_job_due_dates_from_df(
            df=df_jobs_times_final,
            job_column="Job",
            due_date_column="Due Date"
        )

        # Transition Times - ohne Transportzeiten & Liegezeit (Warten auf Halbfabrikate)--------------------------------

        waiting_df = finished_operations.to_waiting_time_dataframe()
        df_avg_waiting = DataFrameEnrichment.aggregate_mean_per_group(
            waiting_df,
            group_column="Machine",
            value_column="Waiting Time",
            new_column_name="Ø Waiting Time",
        )

        MachineInstanceInitializer.insert_from_dataframe(
            df=df_avg_waiting,
            source_name=source_name,
            max_bottleneck_utilization=Decimal(f"{max_bottleneck_utilization}"),
            average_transition_time_column= "Ø Waiting Time"
        )







