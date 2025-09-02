from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Modul 1: Startdeviation (Bars)
# -------------------------------

def prepare_start_deviation_series(
    df_dev: pd.DataFrame,
    experiment_id: Optional[int] = None,
    shift_column: str = "Shift",
    deviation_column: str = "Deviation",
    experiment_column: str = "Experiment_ID",
) -> Dict:
    """
    Liefert die Balken-Serie (Startzeitabweichung je Schicht) für EIN Experiment.
    Rückgabe: dict(shifts: List, values: np.ndarray, data_max: float, exp_used: int)
    """
    if df_dev.empty:
        raise ValueError("df_dev ist leer.")

    if experiment_column not in df_dev.columns:
        raise ValueError(f"Spalte '{experiment_column}' fehlt in df_dev.")

    exps = df_dev[experiment_column].dropna().unique()
    if len(exps) == 0:
        raise ValueError(f"Keine Werte in Spalte '{experiment_column}' in df_dev gefunden.")
    exp_used = experiment_id if experiment_id is not None else exps[0]

    df_e = df_dev.loc[df_dev[experiment_column] == exp_used, [shift_column, deviation_column]].copy()
    if df_e.empty:
        raise ValueError(f"Kein Eintrag für {experiment_column}={exp_used} in df_dev.")

    # Falls mehrere Zeilen je Shift existieren → zur Summe aggregieren
    df_e = (
        df_e.groupby(shift_column, as_index=False)
            .agg({deviation_column: "sum"})
            .sort_values(shift_column)
    )

    shifts = df_e[shift_column].tolist()
    values = df_e[deviation_column].to_numpy(dtype=float)
    data_max = float(values.max()) if len(values) else 0.0

    return dict(shifts=shifts, values=values, data_max=data_max, exp_used=exp_used)


def draw_start_deviation_bars(
    shifts: List,
    values: np.ndarray,
    ax: Optional[plt.Axes] = None,
    label: str = "Durchschnittliche Startzeitenabweichung",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Zeichnet NUR die Balken auf ax (oder erstellt eins) und gibt (fig, ax) zurück.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure  # wichtig: immer ein gültiges Figure-Objekt zurückgeben

    x = np.arange(len(shifts))
    ax.bar(x, values, width=0.8, label=label)
    ax.set_xticks(x, shifts)
    return fig, ax



# Module call -------------------------------------------------------------------------------------------------------

def make_fig_startdeviation_only(
    df_dev: pd.DataFrame,
    experiment_id: int | None = None,
    shift_column: str = "Shift",
    deviation_column: str = "Deviation",
    experiment_column: str = "Experiment_ID",
    y_max: float | None = None,
    y_step: int = 60,
    title: str = "Startzeitabweichungen",
    label: str = "Durchschnittliche Startzeitenabweichung",
):
    spec = prepare_start_deviation_series(df_dev, experiment_id, shift_column, deviation_column, experiment_column)
    fig, ax = draw_start_deviation_bars(spec["shifts"], spec["values"], None, label)
    top = (y_max if y_max is not None else spec["data_max"]) + 15
    if top <= 0: top = 15
    ax.set_xlabel("Schicht (Shift)")
    ax.set_ylabel("Zeitabweichung")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(-15, top)
    ax.axhline(0, linewidth=0.8, color="black")
    import numpy as np
    ax.set_yticks(np.arange(0, top + 1, y_step))
    ax.legend()
    fig.tight_layout()
    return fig
