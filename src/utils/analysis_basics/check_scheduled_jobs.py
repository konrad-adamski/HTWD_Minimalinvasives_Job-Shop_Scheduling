import pandas as pd


def get_jobs_with_lateness_metrics(df_plan_in: pd.DataFrame) -> pd.DataFrame:
    """
    Gibt für jeden Job die letzte Operation zurück und ergänzt Lateness, Tardiness und Earliness.

    Parameter:
    df_plan_in (pd.DataFrame): DataFrame mit Spalten 'Job', 'Operation', 'End', 'Deadline'

    Rückgabe:
    pd.DataFrame: Gefilterter und erweiterter DataFrame mit Lateness-Metriken
    """
    # 1. Letzte Operation je Job selektieren
    df = df_plan_in.sort_values(['Job', 'Operation']).drop_duplicates('Job', keep='last').copy()

    # 2. Lateness-Metriken berechnen
    df["Lateness"] = df["End"] - df["Deadline"]
    df["Tardiness"] = df["Lateness"].clip(lower=0)
    df["Earliness"] = (-df["Lateness"]).clip(lower=0)

    return df