from models import *

# Utils
import src.utils.presenter as show

# Solver Model
from src.models.cp import lateness_scheduling as cp_late_schedule
from src.models.cp import lateness_rescheduling as cp_late_reschedule

# Simulation
from src.simulation.ProductionSimulation import ProductionSimulation

def get_jssp_for_jobs(df_jobs_times, df_routings):
    """
    Erstellt ein JSSP-kompatibles DataFrame durch Verknüpfung von Job-Zeitdaten mit Routings.

    Parameter:
    - df_jobs_times: DataFrame mit mindestens den Spalten 'Job' und 'Routing_ID'.
    - df_routings: DataFrame mit Spalte 'Routing_ID' und den zugehörigen Operationsdaten.

    Rückgabe:
    - df_jssp: DataFrame mit allen für das JSSP notwendigen Informationen, inklusive 'Job' und den Operationen.
    """
    # 1. Relevante Spalten extrahieren
    df_job_ids = df_jobs_times[['Job', 'Routing_ID']].copy()

    # 2. Merge mit df_routings über Routing_ID
    df_jssp = df_job_ids.merge(df_routings, on='Routing_ID')

    return df_jssp

def init_new_version(base_version="base", new_version="new_version"):
    Job.clone_jobs(referenced_version=base_version, new_version=new_version)
    df_jobs_times = Job.get_dataframe(version=new_version)

    # Routings
    df_routings = RoutingOperation.get_dataframe()

    df_jssp = get_jssp_for_jobs(df_jobs_times, df_routings)
    JobOperation.add_from_dataframe(df_jssp, version=new_version, status="open")
    
    
def get_prev_schedule(this_version:str, current_day_numb=1):
    ## Hole den vorherigen Schedule
    df_schedule = Schedule.get_schedule_as_dataframe(date=current_day_numb - 1, version=this_version)

    # Hole die offenen Operationen für die enthaltenen Jobs
    df_open_ops = JobOperation.get_dataframe(version=this_version, jobs=df_schedule["Job"].unique().tolist(),
                                          status="open")

    # Mache ein Inner Join auf ["Job", "Operation"]
    df_schedule_prev = df_schedule.merge(df_open_ops[["Job", "Operation"]], on=["Job", "Operation"], how="inner")
    return df_schedule_prev


def roll_one_day(df_routings:pd.DataFrame, this_version: str,
                     day_numb: int = 1, day_length: int = 1440,
                     max_scheduling_time= 1200):
    day_start = day_length * day_numb
    day_end = day_start + day_length
    print(f"Tag {day_numb:02d}: [{day_start}, {day_end})")

    # alle aktuelle "offenen" Jobs
    df_job_times_curr = Job.get_dataframe(version=this_version, arrival_time_max=day_start, status="open")

    #  JSSP zu allen "offenen" Jobs
    df_jssp_temp = get_jssp_for_jobs(df_job_times_curr, df_routings)
    #  JSSP zu allen "offenen" Jobs, mit "offenen" Operationen
    df_jobs_ops = JobOperation.get_dataframe(version=this_version, jobs=df_jssp_temp.Job.tolist(), status="open")
    df_jssp_curr = df_jssp_temp.merge(df_jobs_ops[['Job', 'Operation']], on=['Job', 'Operation'], how='inner')

    # --- Scheduling ---
    if day_numb == 1:
        print("init scheduling ...")
        df_schedule = cp_late_schedule.solve_jssp_sum_by_tardiness_and_earliness(df_jssp_curr, df_job_times_curr,
                                                                                 schedule_start=day_start, w_t=5,
                                                                                 msg=False,
                                                                                 timeLimit=max_scheduling_time)
    else:
        print(f"scheduling for {day_numb:02d} ...")
        df_ops_in_progess = JobOperation.get_dataframe(version=this_version, status="in progress")
        df_schedule_prev = get_prev_schedule(this_version, day_numb)

        df_schedule = cp_late_reschedule.solve_jssp_advanced(df_jssp_curr, df_job_times_curr, df_ops_in_progess,
                                                             df_original_plan=df_schedule_prev, w_t=5,
                                                             r=0.2,  # 20% Lateness, 80% Deviation
                                                             reschedule_start=day_start,
                                                             msg=False, timeLimit=max_scheduling_time, gapRel=0.001,
                                                             alpha=0.90)

    json_schedule = df_schedule.to_dict(orient='records')
    Schedule.add_schedule(data=json_schedule, date=day_numb, version=this_version)
    show.plot_gantt(df_schedule, perspective="Machine", title=f"Gantt-Diagramm ab Tag {day_numb}")

    # --- Simulation ---
    execute_one_day(df_schedule, this_version, day_numb)


def execute_one_day(df_schedule: pd.DataFrame, this_version: str, day_numb: int = 1, day_length: int = 1440):
    day_start = day_length * day_numb
    day_end = day_start + day_length

    # Simulation
    simulation = ProductionSimulation(df_schedule, sigma=0.2)
    df_execution = simulation.run(start_time=day_start, end_time=day_end)
    show.plot_gantt(df_execution, perspective="Machine", title=f"Gantt-Diagramm für Tag {day_numb}")

    # previous "in progress operations" -> "finished"
    df_ops_in_progess_prev = JobOperation.get_dataframe(version=this_version, status="in progress")
    if df_ops_in_progess_prev is not None and not df_ops_in_progess_prev.empty:
        df_ops_in_progess_prev_finished = df_ops_in_progess_prev[df_ops_in_progess_prev.End <= day_end]
        JobOperation.add_from_dataframe(df_ops_in_progess_prev_finished, version=this_version, status="finished")

    # finished operations from current simulation
    df_ops_finished = df_execution[df_execution.End < day_end]
    JobOperation.add_from_dataframe(df_ops_finished, version=this_version, status="finished")
    JobOperation.update_closed_jobs_from_operations(version=this_version)

    # operations in progress from current simulation
    df_ops_in_progess = df_execution[df_execution.End >= day_end]
    JobOperation.add_from_dataframe(df_ops_in_progess, version=this_version, status="in progress")
    
if __name__ == "__main__":
    this_version = "test_version"
    max_scheduling_time = 60*10 # 10 min

    init_new_version(base_version="base", new_version=this_version)
    # Routings
    df_routings = RoutingOperation.get_dataframe()

    last_planning_start = 2
    for day_numb in range(1, last_planning_start + 1):
        roll_one_day(df_routings=df_routings, this_version=this_version,
                     day_numb=day_numb, max_scheduling_time=max_scheduling_time)
    
    
    
    