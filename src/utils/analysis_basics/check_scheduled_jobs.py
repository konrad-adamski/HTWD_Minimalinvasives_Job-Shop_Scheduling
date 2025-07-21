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

# Dataframe gruppiert ----------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------

def count_column_grouped(df: pd.DataFrame, column: str = 'Tardiness', steps = 60, min_val = 0, max_val= 180, right_closed: bool = False) -> pd.Series:

    # 1. Bins je nach Spalte
    if column in ['Tardiness', "Absolute Lateness"]:
        inner_bins = list(range(min_val, max_val + steps, steps))
        bins = inner_bins + [np.inf]  # Kein -inf
    else:
        min_val = -max_val if min_val >= 0 else min_val
        inner_bins = list(range(min_val, max_val + steps, steps))
        inner_bins = list(range(min_val, max_val + steps, steps))
        if 0 not in inner_bins:
            inner_bins.append(0)
            inner_bins = sorted(inner_bins)
        bins = [-np.inf] + inner_bins + [np.inf]  # Mit -inf und +inf


    # 2. Spalte prüfen
    if column not in df.columns:
        raise ValueError(f"Spalte '{column}' existiert nicht. Verfügbare Spalten: {list(df.columns)}")

    # 3. Zähle Null-Werte separat
    zero_count = (df[column] == 0).sum()
    non_zero = df.loc[df[column] != 0, column]

    # 4. Labels erzeugen
    labels = []
    bin_keys = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if np.isneginf(lo):
            labels.append(f"<{int(hi)}")
            bin_keys.append(lo + 0.1)  # z.B. -119.9
        elif np.isposinf(hi):
            labels.append(f">{int(lo)}")
            bin_keys.append(hi - 0.1)  # z.B. 2880 - 0.1
        else:
            labels.append(f"{int(lo)} - {int(hi)}")
            bin_keys.append((lo + hi) / 2)

    # 5. Cutting
    grouped = pd.cut(non_zero, bins=bins, labels=labels, right=right_closed, include_lowest=True)
    counts = grouped.value_counts().reindex(labels, fill_value=0)

    # 6. Zero-Label einfügen
    counts["0"] = zero_count
    bin_keys.append(0)  # Füge Schlüssel für '0' ein
    labels.append("0")

    # 7. Korrekt sortieren
    sort_df = pd.DataFrame({'label': labels, 'key': bin_keys}).set_index('label')
    sorted_counts = counts.loc[sort_df.sort_values('key').index]

    return sorted_counts.astype(int)