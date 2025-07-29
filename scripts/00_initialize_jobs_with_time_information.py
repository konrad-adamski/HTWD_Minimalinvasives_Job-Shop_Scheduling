import pandas as pd

from src.utils.editor import enrich_schedule_dframe
# Utils
from src.utils.initialization import jobs_jssp_init as init
from src.utils.initialization.gen_deadlines import get_temporary_df_times_from_schedule, \
    add_groupwise_lognormal_deadlines_by_group_mean, ensure_reasonable_deadlines

#Data access
from configs.path_manager import get_path

# Simulation
from src.simulation.ProductionRollingSimulation import ProductionSimulation


if __name__ == '__main__':
    # params
    u_b_mmax = 0.90
    lognormal_sigma = 0.25

    basic_data_path = get_path("data", "basic")
    df_routings = pd.read_csv(basic_data_path / "ft10_routings.csv")


    # --- JSSP and arrivals ---
    day_count = 360
    df_jssp, df_jobs_arrivals = init.create_jobs_for_shifts(
        df_routings=df_routings,
        shift_count=day_count,
        shift_length=1440,
        u_b_mmax=u_b_mmax,
        shuffle=True,
        job_seed= 50,
        arrival_seed= 122
    )

    # --- FCFS Simulation ---
    df_problem = df_jssp.merge(
        df_jobs_arrivals[['Job', 'Routing_ID', 'Arrival', 'Ready Time']],
        on=['Job', 'Routing_ID'],
        how='left'
    )

    simulation = ProductionSimulation(earliest_start_column="Ready Time", sigma=0, verbose=False)
    simulation.run(df_problem, start_time=0, end_time=None)
    df_fcfs_execution = simulation.get_finished_operations_df()
    df_pseudo_schedule = enrich_schedule_dframe(df_fcfs_execution, df_jobs_arrivals)

    # --- Prepare job-level timing summary for routing-based deadline generation ---
    df_jobs_times_temp = get_temporary_df_times_from_schedule(df_pseudo_schedule, df_jssp)

    df_jobs_times = add_groupwise_lognormal_deadlines_by_group_mean(df_jobs_times_temp, sigma=lognormal_sigma)
    df_jobs_times_final = ensure_reasonable_deadlines(df_jobs_times, min_coverage = 1.0)

    df_jobs_times_final.to_csv(basic_data_path / f"ft10_jobs_times.csv", index=False)