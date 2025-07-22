import numpy as np
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

# Dataframe gruppiert --------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

def get_jobs_aggregated(df: pd.DataFrame, column: str = 'Tardiness', steps=60, min_val=0, max_val=120, right_closed: bool = False) -> pd.DataFrame:
    # 1. Spalte prüfen
    if column not in df.columns:
        raise ValueError(f"Spalte '{column}' existiert nicht. Verfügbare Spalten: {list(df.columns)}")

    # 2. Bins definieren
    if column in ['Tardiness', "Absolute Lateness"]:
        inner_bins = list(range(min_val, max_val + steps, steps))
        bins = inner_bins + [np.inf]  # kein -inf
    else:
        min_val = -max_val if min_val >= 0 else min_val
        inner_bins = list(range(min_val, max_val + steps, steps))
        if 0 not in inner_bins:
            inner_bins.append(0)
            inner_bins = sorted(inner_bins)
        bins = [-np.inf] + inner_bins + [np.inf]  # mit -inf und +inf

    # 3. Null-Werte separat zählen
    zero_count = (df[column] == 0).sum()
    non_zero = df.loc[df[column] != 0, column]

    # 4. Labels & Sortierschlüssel
    labels = []
    bin_keys = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if np.isneginf(lo):
            labels.append(f"<{int(hi)}")
            bin_keys.append(lo + 0.1)
        elif np.isposinf(hi):
            labels.append(f">{int(lo)}")
            bin_keys.append(hi - 0.1)
        else:
            labels.append(f"{int(lo)} - {int(hi)}")
            bin_keys.append((lo + hi) / 2)

    # 5. Cutting der Nicht-Null-Werte
    grouped = pd.cut(non_zero, bins=bins, labels=labels, right=right_closed, include_lowest=True)
    counts = grouped.value_counts().reindex(labels, fill_value=0)

    # 6. Zero-Werte ergänzen
    counts["0"] = zero_count
    bin_keys.append(0)
    labels.append("0")

    # 7. Nach Sortierschlüssel ordne
    sort_df = pd.DataFrame({f'{column}_Intervall': labels, 'key': bin_keys}).set_index(f'{column}_Intervall')
    sorted_counts = counts.loc[sort_df.sort_values('key').index]

    # 8. Ausgabe als DataFrame mit einer Zeile, Intervallnamen als Spalten
    return pd.DataFrame([sorted_counts])


