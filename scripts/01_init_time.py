from models import *

# Utils
from src.utils.production_process import initialize as init

# Simulation
from src.simulation.ProductionSimulation import ProductionSimulation

# External
import pandas as pd
import numpy as np

def add_groupwise_lognormal_deadlines_by_group_mean(df: pd.DataFrame, sigma: float = 0.2,
                                                    routing_column: str = "Routing_ID", seed: int = 42) -> pd.DataFrame:
    """
    Für jede Gruppe in 'Routing_ID' wird eine Lognormalverteilung
    mit Parameter mu so berechnet, dass der Mittelwert der Deadlines genau
    dem Mittelwert der 'End'-Werte der Gruppe entspricht.

    Jeder Deadline-Wert in der Gruppe wird einzeln zufällig aus dieser Verteilung gezogen.

    Parameters
    ----------
    df : pd.DataFrame
        Muss Spalten routing_column und 'End' enthalten.
    sigma : float, optional
        Standardabweichung der Lognormalverteilung (Default 0.2).
    seed : int
        Zufalls-Seed (Default 42).

    Returns
    -------
    pd.DataFrame
        Kopie von df mit neuer Spalte 'Deadline' (float, 1 Dezimalstelle).
    """
    np.random.seed(seed)
    df_out = df.copy()
    df_out['Deadline'] = np.nan

    for routing_id, grp in df_out.groupby(routing_column):
        target_flow_mean = grp['End'].mean() - grp['Ready Time'].mean()
        mu = np.log(target_flow_mean) - 0.5 * sigma**2

        # Für jede Zeile in Gruppe eine Deadline aus LogNormal(mu, sigma)
        flow_budgets = np.random.lognormal(mean=mu, sigma=sigma, size=len(grp))
        df_out.loc[grp.index, 'Deadline'] = df_out.loc[grp.index, 'Ready Time'] + np.round(flow_budgets)

    return df_out


if __name__ == "__main__":
    # Routings
    df_routings = RoutingOperation.get_dataframe()

    day_count = 200
    export_day_count = 31

    df_jssp, df_jobs_arrivals = init.create_jobs_for_shifts(df_routings=df_routings,
                                                            routing_column="Routing_ID", job_column="Job",
                                                            shift_count=day_count, shift_length=1440,
                                                            u_b_mmax=0.92, shuffle=True
                                                            )

    # --- Simulation ---
    df_scheduling_problem = df_jssp.merge(
        df_jobs_arrivals[['Job', 'Routing_ID', 'Arrival', 'Ready Time']],
        on=['Job', 'Routing_ID'],
        how='left'
    )

    simulation = ProductionSimulation(df_scheduling_problem, earliest_start_column="Ready Time", verbose=False, sigma=0)
    df_fcfs_execution = simulation.run(start_time=0, end_time=None)

    # Letzte Operation je Job auswählen
    df_last_ops = df_fcfs_execution.sort_values("Operation").groupby("Job").last().reset_index()
    df_jobs_times = df_last_ops[["Job", "Routing_ID", "Arrival", "Ready Time", "End"]]

    # Gesamtbearbeitungszeit
    df_proc_time = df_jssp.groupby("Job", as_index=False)["Processing Time"].sum()

    # Merge
    df_jobs_times = df_jobs_times.merge(df_proc_time, on="Job", how="left")


    # --- "lognormal" Deadlines ----

    df_times = add_groupwise_lognormal_deadlines_by_group_mean(df_jobs_times, sigma=0.3)


    # untere Grenze
    df_times['Deadline'] = np.maximum(df_times['Deadline'],
                                      df_times['Ready Time']
                                      + df_times['Processing Time'] / 2
                                     )
    df_times['Deadline'] = np.ceil(df_times['Deadline']).astype(int)


    # Einfügen in die Datenbank (31 Tage)
    df_times_export = df_times[df_times['Ready Time'] < 60*24*export_day_count]
    print(df_times_export.head(10))
    Job.add_jobs_from_dataframe(df_times_export, version="base", due_date_column='Deadline')



