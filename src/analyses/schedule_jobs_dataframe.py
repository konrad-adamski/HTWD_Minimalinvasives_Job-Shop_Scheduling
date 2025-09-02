from typing import Literal

import pandas as pd


def jobs_metrics_from_operations_df(
    df_ops: pd.DataFrame,
    job_column: str = "Job",
    routing_column: str = "Routing_ID",
    experiment_column: str = "Experiment_ID",
    shift_column: str = "Shift",
    arrival_column: str = "Arrival",
    due_date_column: str = "Due Date",
    operation_column: str = "Operation",
    end_column: str = "End",
    tardiness_column: str = "Tardiness",
    earliness_column: str = "Earliness",
    lateness_column: str = "Lateness",
    completion_column: str = "Completion",
) -> pd.DataFrame:
    """
    Nimmt das Operations-DataFrame und wählt für jeden Job (innerhalb eines Experiments)
    die Zeile mit der höchsten Operation-Nummer.
    Daraus werden Completion, Lateness, Tardiness, Earliness berechnet.
    """
    # letzte Operation je Job *innerhalb eines Experiments* finden
    idx = df_ops.groupby([experiment_column, job_column])[operation_column].idxmax()
    df_last = df_ops.loc[idx].copy()

    # Completion = End der letzten Operation
    df_last[completion_column] = df_last[end_column]

    # Kennzahlen
    df_last[lateness_column] = df_last[completion_column] - df_last[due_date_column]
    df_last[tardiness_column] = df_last[lateness_column].clip(lower=0)
    df_last[earliness_column] = (-df_last[lateness_column]).clip(lower=0)

    # gewünschte Spalten
    ordered_cols = [
        job_column,
        routing_column,
        experiment_column,
        shift_column,
        arrival_column,
        due_date_column,
        completion_column,
        tardiness_column,
        earliness_column,
        lateness_column,
    ]

    return df_last[ordered_cols].sort_values([experiment_column, job_column], ignore_index=True)

def mean_start_deviation_per_shift_df(
    df_ops: pd.DataFrame,
    job_column: str = "Job",
    operation_column: str = "Operation",
    shift_column: str = "Shift",
    start_column: str = "Start",
    experiment_column: str = "Experiment_ID",
) -> pd.DataFrame:

    return _calculate_start_deviation_per_shift_df(
        df_ops=df_ops,
        method = "mean",
        job_column=job_column,
        operation_column=operation_column,
        shift_column=shift_column,
        start_column=start_column,
        experiment_column=experiment_column,
    )

def _calculate_start_deviation_per_shift_df(
    df_ops: pd.DataFrame,
    method: Literal["mean", "sum"] = "mean",
    job_column: str = "Job",
    operation_column: str = "Operation",
    shift_column: str = "Shift",
    start_column: str = "Start",
    experiment_column: str = "Experiment_ID",
) -> pd.DataFrame:
    """
    Berechnet pro Schicht die Startzeitenabweichung zur VORHERIGEN Schicht.
    - match: (Experiment_ID, Job, Operation)
    - deviation = |Start_new - Start_orig|
    - method: "sum" oder "mean"
    Return: DataFrame mit Spalten [Experiment_ID, Shift, Deviation, Pairs]
    """
    if method not in {"sum", "mean"}:
        raise ValueError("method must be 'sum' or 'mean'.")

    shifts = sorted(df_ops[shift_column].unique())
    if len(shifts) < 2:
        return pd.DataFrame(columns=[experiment_column, shift_column, "Deviation", "Pairs"])

    experiments = sorted(df_ops[experiment_column].unique())
    rows = []

    for s_idx in range(1, len(shifts)):
        s_prev = shifts[s_idx - 1]
        s_curr = shifts[s_idx]

        for exp in experiments:
            df_prev = df_ops.loc[
                (df_ops[shift_column] == s_prev) & (df_ops[experiment_column] == exp),
                [experiment_column, job_column, operation_column, start_column]
            ].rename(columns={start_column: "Start_orig"})

            df_curr = df_ops.loc[
                (df_ops[shift_column] == s_curr) & (df_ops[experiment_column] == exp),
                [experiment_column, job_column, operation_column, start_column]
            ].rename(columns={start_column: "Start_new"})

            merged = pd.merge(
                df_curr,
                df_prev,
                on=[experiment_column, job_column, operation_column],
                how="inner",
            )

            if merged.empty:
                rows.append({experiment_column: exp, shift_column: s_curr, "Deviation": 0.0, "Pairs": 0})
                continue

            merged["Deviation"] = (merged["Start_new"] - merged["Start_orig"]).abs()
            dev = merged["Deviation"].sum() if method == "sum" else merged["Deviation"].mean()
            rows.append({experiment_column: exp, shift_column: s_curr, "Deviation": float(dev), "Pairs": int(len(merged))})

    df_dev = pd.DataFrame(rows).sort_values([experiment_column, shift_column], ignore_index=True)
    return df_dev