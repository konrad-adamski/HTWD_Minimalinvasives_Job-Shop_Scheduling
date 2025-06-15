import pandas as pd


# "Lateness" (Tardiness und Earliness) ------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
def compute_tardiness_earliness_ideal_ratios(df_plan_last_ops_list):
    """
    Berechnet für jeden Tag den Anteil an:
    - Tardiness > 0
    - Earliness > 0
    - Ideal (T=0 & E=0)

    Gibt NaN zurück, wenn ein DataFrame leer ist.
    """
    tardiness_ratio_per_day = []
    earliness_ratio_per_day = []
    ideal_ratio_per_day = []

    for df in df_plan_last_ops_list:
        tardiness_ratio = (df["Tardiness"] > 0).mean()
        earliness_ratio = (df["Earliness"] > 0).mean()
        ideal_ratio = ((df["Tardiness"] == 0) & (df["Earliness"] == 0)).mean()

        tardiness_ratio_per_day.append(tardiness_ratio)
        earliness_ratio_per_day.append(earliness_ratio)
        ideal_ratio_per_day.append(ideal_ratio)

    return tardiness_ratio_per_day, earliness_ratio_per_day, ideal_ratio_per_day

def compute_mean_tardiness_earliness(df_plan_last_ops_list):
    mean_tardiness_per_day = []
    mean_earliness_per_day = []

    for df in df_plan_last_ops_list:
        mean_tardiness = df["Tardiness"].mean()
        mean_earliness = df["Earliness"].mean()

        mean_tardiness_per_day.append(mean_tardiness)
        mean_earliness_per_day.append(mean_earliness)

    return mean_tardiness_per_day, mean_earliness_per_day


def compute_nonzero_mean_tardiness_earliness(df_plan_last_ops_list):
    mean_tardiness_per_day = []
    mean_earliness_per_day = []

    for df in df_plan_last_ops_list:
        # Nur Tardiness-Werte > 0 berücksichtigen
        tardiness_values = df["Tardiness"][df["Tardiness"] > 0]
        if not tardiness_values.empty:
            mean_tardiness = tardiness_values.mean()
        else:
            mean_tardiness = 0.0
        mean_tardiness_per_day.append(mean_tardiness)

        # Nur Earliness-Werte > 0 berücksichtigen
        earliness_values = df["Earliness"][df["Earliness"] > 0]
        if not earliness_values.empty:
            mean_earliness = earliness_values.mean()
        else:
            mean_earliness = 0.0
        mean_earliness_per_day.append(mean_earliness)

    return mean_tardiness_per_day, mean_earliness_per_day

# Deviation ---------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------

def compute_daily_starttime_deviations(plan_list, method="sum", with_T1=True):
    """
    Berechnet die tägliche Startzeit-Abweichung zwischen aufeinanderfolgenden Plänen.

    - Tag 0: Deviation = 0
    - Ab Tag 1: Vergleich mit jeweils vorherigem Tag

    Parameter:
    - plan_list: Liste von DataFrames mit Spalten für Job, Operation und Startzeit
    - method: "sum" für Gesamtabweichung, "mean" für durchschnittliche Abweichung pro Operation

    Rückgabe:
    - Liste der täglichen Abweichungen (float)
    """
    deviations = [0.0]  # Tag 0 ist Referenz

    for i in range(1, len(plan_list)):
        if with_T1:
            deviation = calculate_deviation_after_T1(df_original=plan_list[i - 1], df_new=plan_list[i], method=method)
        else:    
            deviation = calculate_deviation_wu(df_original=plan_list[i - 1], df_new=plan_list[i], method=method)
        deviations.append(deviation)

    return deviations
    
def calculate_deviation_wu(df_original, df_new, method="sum"):
    """
    Berechnet die Abweichung der Startzeiten zwischen ursprünglichem und neuem Plan.

    Parameter:
    - method: "sum" für Gesamtabweichung, "mean" für durchschnittliche Abweichung

    Rückgabe:
    - float: gewünschte Abweichung (Summe oder Mittelwert)
    """
    # Merge aller Operationen
    merged = pd.merge(
        df_new[["Job", "Operation", "Start"]],
        df_original[["Job", "Operation", "Start"]],
        on=["Job", "Operation"],
        suffixes=('_new', '_orig')
    )

    merged["Deviation"] = (merged["Start_new"] - merged["Start_orig"]).abs()

    if method == "mean":
        return merged['Deviation'].mean()
    else:
        return merged['Deviation'].sum()


def calculate_deviation_after_T1(df_original, df_new, method="sum"):
    """
    Berechnet die Abweichung der Startzeiten zwischen ursprünglichem und neuem Plan,
    beginnend ab dem abgerundeten T1-Wert aus df_new.

    Parameter:
    - method: "sum" für Gesamtabweichung, "mean" für durchschnittliche Abweichung

    Rückgabe:
    - float: gewünschte Abweichung (Summe oder Mittelwert)
    """
    T1 = get_T1(df_new)

    # Ursprüngliche Daten auf Operationen ab T1 eingrenzen
    df_original_filtered = df_original[df_original["Start"] >= T1]

    # Merge nur auf relevante Operationen
    merged = pd.merge(
        df_new[["Job", "Operation", "Start"]],
        df_original_filtered[["Job", "Operation", "Start"]],
        on=["Job", "Operation"],
        suffixes=('_new', '_orig')
    )

    merged["Deviation"] = (merged["Start_new"] - merged["Start_orig"]).abs()

    if method == "mean":
        return merged["Deviation"].mean()
    else:
        return merged["Deviation"].sum()


def get_T1(df_revised, base=1440):
    """
    Bestimmt den kleinsten Startzeitpunkt im DataFrame df_revised und rundet ihn auf das nächstkleinere Vielfache von `base` ab.

    Parameter:
    - df_revised: pandas.DataFrame mit einer Spalte 'Start'
    - base: Ganzzahl, auf deren Vielfaches abgerundet wird (Standard: 1440)

    Rückgabe:
    - int: Abgerundeter Wert des kleinsten Startzeitpunkts
    """
    min_start = df_revised["Start"].min()
    return (min_start // base) * base

        