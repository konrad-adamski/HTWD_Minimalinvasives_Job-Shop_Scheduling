# Datenzugriff
from configs.path_manager import get_path

# Utils
from src.utils.initialization import jobs_jssp_init as init
from src.utils.initialization.gen_deadlines import get_temporary_df_times_from_schedule, add_groupwise_lognormal_deadlines_by_group_mean, improve_created_deadlines

# Simulation
from src.simulation.ProductionRollingSimulation import ProductionSimulation

# Extern
import pandas as pd


if __name__ == '__main__':
    basic_data_path = get_path("data", "basic")
    df_routings = pd.read_csv(basic_data_path / "instance.csv")

    day_count = 360
    df_jssp, df_jobs_arrivals = init.create_jobs_for_shifts(df_routings=df_routings,
                                                            routing_column="Routing_ID", job_column="Job",
                                                            shift_count=day_count, shift_length=1440,
                                                            u_b_mmax=0.90, shuffle=True
                                                            )


    # --- FCFS Simulation ---
    df_problem = df_jssp.merge(
        df_jobs_arrivals[['Job', 'Routing_ID', 'Arrival', 'Ready Time']],
        on=['Job', 'Routing_ID'],
        how='left'
    )
    df_problem

    simulation = ProductionSimulation(earliest_start_column="Ready Time", sigma=0, verbose=False)
    simulation.run(df_problem, start_time=0, end_time=None)
    df_fcfs_execution = simulation.get_finished_operations_df()

    # --- temporäre Produktionsauftragsdaten ---
    df_jobs_times_temp = get_temporary_df_times_from_schedule(df_fcfs_execution, df_jssp)

    # --- Deadlines aus Log-Normalverteilung der FlowTime/Slack ---
    df_times = add_groupwise_lognormal_deadlines_by_group_mean(df_jobs_times_temp, sigma=0.3)

    df_times = improve_created_deadlines(df_times, min_covered_proc_times_percentage=0.75)

    # Export
    basic_data_path = get_path("data", "basic")
    df_times.to_csv(basic_data_path / f"jobs_times_final.csv", index=False)
    df_jssp.to_csv(basic_data_path / f"jssp_final.csv", index=False)








