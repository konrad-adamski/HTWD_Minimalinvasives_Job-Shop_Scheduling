import pandas as pd
import numpy as np

def get_temporary_df_times_from_schedule(df_schedule: pd.DataFrame, df_jssp: pd.DataFrame) -> pd.DataFrame:
    # Letzte Operation je Job auswählen
    df_last_ops = df_schedule.sort_values("Operation").groupby("Job").last().reset_index()
    df_jobs_times = df_last_ops[["Job", "Routing_ID", "Arrival", "Ready Time", "End"]]

    # Gesamtbearbeitungszeit
    df_proc_time = df_jssp.groupby("Job", as_index=False)["Processing Time"].sum()
    df_proc_time.rename(columns={"Processing Time": "Job Processing Time"}, inplace=True)

    # Merge
    df_jobs_times = df_jobs_times.merge(df_proc_time, on="Job", how="left")
    df_jobs_times

    return df_jobs_times

def add_groupwise_lognormal_deadlines_by_group_mean(df_times_temp: pd.DataFrame, sigma: float = 0.2,
                                                    routing_column: str = "Routing_ID", seed: int = 42) -> pd.DataFrame:
    """
    Für jede Gruppe in 'Routing_ID' wird eine Lognormalverteilung
    mit Parameter mu so berechnet, dass der Mittelwert der Deadlines genau
    dem Mittelwert der 'End'-Werte der Gruppe entspricht.

    Jeder Deadline-Wert in der Gruppe wird einzeln zufällig aus dieser Verteilung gezogen.

    Parameters
    ----------
    df_times_temp : pd.DataFrame
        Muss Spalten routing_column und 'End' enthalten.
    sigma : float, optional
        Standardabweichung der Lognormalverteilung (Default 0.2).
    seed : int
        Zufalls-Seed (Default 42).

    Returns
    -------
    pd.DataFrame
        Kopie von df_times_temp mit neuer Spalte 'Deadline'.
    """
    np.random.seed(seed)
    df_times = df_times_temp.copy()
    df_times['Deadline'] = np.nan

    for routing_id, grp in df_times.groupby(routing_column):
        target_flow_mean = grp['End'].mean() - grp['Ready Time'].mean()
        mu = np.log(target_flow_mean) - 0.5 * sigma**2

        # Für jede Zeile in Gruppe eine Deadline aus LogNormal(mu, sigma)
        flow_budgets = np.random.lognormal(mean=mu, sigma=sigma, size=len(grp))
        df_times.loc[grp.index, 'Deadline'] = df_times.loc[grp.index, 'Ready Time'] + np.round(flow_budgets)

    return df_times


def improve_created_deadlines(df_times: pd.DataFrame, min_covered_proc_times_percentage: float = 0.75) -> pd.DataFrame:
    min_covered_proc_times_percentage = min(min_covered_proc_times_percentage, 1)
    df_times['Deadline'] = np.maximum(df_times['Deadline'],
                                      df_times['Ready Time']
                                      + df_times['Job Processing Time'] * min_covered_proc_times_percentage
                                      )

    df_times['Deadline'] = np.ceil(df_times['Deadline']).astype(int)
    return df_times