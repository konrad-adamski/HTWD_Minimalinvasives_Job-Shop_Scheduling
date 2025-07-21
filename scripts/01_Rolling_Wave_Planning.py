# Datenzugriff
from configs.path_manager import get_path
import pickle

# Utils
from src.utils.analize import check_constrains as check
from src.utils.rolling_process.rolling_filter import *
import src.utils.presenter as show

# Solver Model
from src.models.cp import lateness_scheduling as cp_late_schedule
from src.models.cp import lateness_rescheduling as cp_late_reschedule

# Simulation
from src.simulation.ProductionRollingSimulation import ProductionSimulation

# Extern
import pandas as pd

pd.set_option('display.min_rows', 12)
pd.set_option('display.max_rows', 16)

basic_data_path = get_path("data", "basic")
shift_length = 1440

def prepare_wave(df_jssp: pd.DataFrame, df_jobs_times: pd.DataFrame,
                 df_not_started: pd.DataFrame | None = None, this_shift_length: int = 1440,
                 day_numb: int = 1, exclusion_dataframes_list: list = []):
    day_start = this_shift_length * day_numb
    day_end = day_start + this_shift_length
    print(f"Tag {day_numb:02d}: [{day_start}, {day_end})")

    # "neue" und unerledigte Jobs
    df_jobs_times_current = get_current_jobs(
        df_jobs_times, df_previous_not_started=df_not_started, ready_time=day_start
    )

    df_jssp_current = filter_current_jssp(
       df_jssp=df_jssp,
       df_jobs_times_current=df_jobs_times_current,
       exclusion_dataframes_list = exclusion_dataframes_list
    )
    return df_jssp_current, df_jobs_times_current

def schedule_init_wave(df_jssp: pd.DataFrame, df_jobs_times: pd.DataFrame,
                       shift_length: int = 1440, max_scheduling_time: int | None = 3600):
    df_jssp_current, df_jobs_times_current = prepare_wave(
        df_jssp=df_jssp, df_jobs_times=df_jobs_times, this_shift_length=shift_length, day_numb = 1
    )

    # --- Scheduling ---
    df_schedule = cp_late_schedule.solve_jssp_sum_by_tardiness_and_earliness(
        df_jssp_current, df_jobs_times_current, earliest_start_column="Ready Time", w_t=5,
        msg=False, timeLimit=max_scheduling_time, gapRel=0.01)

    return df_schedule


def reschedule_wave(df_jssp: pd.DataFrame, df_jobs_times: pd.DataFrame, df_schedule_prev: pd.DataFrame,
                    df_execution_prev: pd.DataFrame, df_active_prev: pd.DataFrame | None,
                    df_not_started: pd.DataFrame | None = None, shift_length: int = 1440,
                    day_numb: int = 2, objective_function_type: str = "simple",
                    max_scheduling_time: int | None = 3600):
    day_start = shift_length * day_numb

    if df_active_prev is not None:
        exclusion_dataframes_list = [df_execution_prev, df_active_prev]
    else:
        exclusion_dataframes_list = [df_execution_prev]

    df_jssp_current, df_jobs_times_current = prepare_wave(
        df_jssp=df_jssp, df_jobs_times=df_jobs_times, df_not_started=df_not_started,
        this_shift_length=shift_length, day_numb=day_numb, exclusion_dataframes_list = exclusion_dataframes_list
    )

    if objective_function_type == "devpen":
        df_schedule_out = cp_late_reschedule.solve_jssp_by_tardiness_and_earliness_with_devpen(
            df_jssp=df_jssp_current, df_times=df_jobs_times_current,
            df_original_plan=df_schedule_prev, df_active=df_active_prev,
            reschedule_start=day_start, w_t=5,
            r=0.30,  # 30% Flowtime, 70% Abweichung
            msg=False, timeLimit=max_scheduling_time, gapRel=0.02
        )
    else: # Simple
        df_schedule_out = cp_late_reschedule.solve_jssp_by_tardiness_and_earliness_with_fixed_ops(
            df_jssp_current, df_jobs_times_current, df_active_prev,
            reschedule_start=day_start, w_t=5,
            msg=False, timeLimit=max_scheduling_time, gapRel=0.02
        )

    return df_schedule_out


def simulate_one_shift(simulation: ProductionSimulation, df_schedule_in: pd.DataFrame,
                       day_numb: int, shift_length:int = 1440):
    day_start = shift_length * day_numb
    day_end = day_start + shift_length

    simulation.run(dframe_schedule_plan=df_schedule_in, start_time=day_start, end_time=day_end)
    df_execution_out = simulation.get_finished_operations_df()
    show.plot_gantt(df_execution_out, perspective="Machine",
                    title=f"Gantt Diagramm für die abgeschlossenen Arbeitsgänge am Tag {day_numb:02d}")

    df_active_out = simulation.get_active_operations_df()

    df_not_started_out = simulation.get_not_started_operations_df(df_schedule_in)

    return df_execution_out, df_active_out, df_not_started_out



if __name__ == "__main__":
    df_jssp = pd.read_csv(basic_data_path / "jssp_final.csv")
    df_jobs_times = pd.read_csv(basic_data_path / "jobs_times_final.csv")

    simulation = ProductionSimulation(sigma=0.15)
    day_length = 1440
    max_scheduling_time = 60*20 # 20 min
    last_planning_start = 3

    # Tag 1 (init)
    df_schedule = schedule_init_wave(
        df_jssp= df_jssp, df_jobs_times=df_jobs_times,
        shift_length=day_length, max_scheduling_time=max_scheduling_time
    )

    # df_schedule.to_csv("", index=False)

    df_execution, df_active, df_not_started = simulate_one_shift(
        simulation=simulation, df_schedule_in=df_schedule, day_numb=1, shift_length=day_length
    )

    # Tag 2+

    for day_numb in range(2, last_planning_start + 1):
        df_schedule = reschedule_wave(
            df_jssp= df_jssp, df_jobs_times=df_jobs_times, df_schedule_prev=df_schedule,df_execution_prev= df_execution,
            df_active_prev= df_active, df_not_started=df_not_started, shift_length=day_length, day_numb=day_numb,
            objective_function_type="simple", max_scheduling_time=max_scheduling_time
        )

        # df_schedule.to_csv("", index=False)

        df_execution, df_active, df_not_started = simulate_one_shift(
            simulation=simulation, df_schedule_in=df_schedule, day_numb=day_numb, shift_length=day_length
        )






