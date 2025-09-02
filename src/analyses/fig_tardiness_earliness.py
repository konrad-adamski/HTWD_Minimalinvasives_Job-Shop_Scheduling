from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------
# Modul 2: Tardiness/Earliness (Lines)
# --------------------------------------

def prepare_avg_metrics_series(
    df_metrics: pd.DataFrame,
    experiment_id: Optional[int] = None,
    shift_column: str = "Shift",
    tardiness_column: str = "Tardiness",
    earliness_column: str = "Earliness",
    experiment_column: str = "Experiment_ID",
    show_earliness: bool = True,
) -> Dict:
    """
    Liefert die Linien-Serien (Ø Tardiness/Earliness je Schicht) für EIN Experiment.
    Rückgabe: dict(shifts: List, tard: np.ndarray, earl: np.ndarray|None, data_max: float, exp_used: int|None)
    """
    if df_metrics.empty:
        raise ValueError("df_metrics ist leer.")

    if experiment_column in df_metrics.columns:
        exps = df_metrics[experiment_column].dropna().unique()
        if len(exps) == 0:
            raise ValueError(f"Keine Werte in Spalte '{experiment_column}' in df_metrics gefunden.")
        exp_used = experiment_id if experiment_id is not None else exps[0]
        df_m = df_metrics.loc[df_metrics[experiment_column] == exp_used].copy()
        if df_m.empty:
            raise ValueError(f"Kein Eintrag für {experiment_column}={exp_used} in df_metrics.")
    else:
        # Falls df_metrics kein Experiment enthält → komplett verwenden
        exp_used = experiment_id
        df_m = df_metrics.copy()

    df_shift = (
        df_m.groupby(shift_column, as_index=False)
            .agg(
                avg_tardiness=(tardiness_column, "mean"),
                avg_earliness=(earliness_column, "mean"),
            )
            .sort_values(shift_column)
    )

    shifts = df_shift[shift_column].tolist()
    tard = df_shift["avg_tardiness"].to_numpy(dtype=float)
    earl = df_shift["avg_earliness"].to_numpy(dtype=float) if show_earliness else None

    if earl is not None:
        data_max = float(np.nanmax([np.nanmax(tard), np.nanmax(earl)]))
    else:
        data_max = float(np.nanmax(tard)) if len(tard) else 0.0

    return dict(shifts=shifts, tard=tard, earl=earl, data_max=data_max, exp_used=exp_used)


def draw_avg_metrics_lines(
    shifts: List,
    tard: np.ndarray,
    earl: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    tardiness_label: str = "Ø Tardiness",
    earliness_label: str = "Ø Earliness",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Zeichnet NUR die Linien (Ø Tardiness/Earliness) auf ax (oder erstellt eins) und gibt (fig, ax) zurück.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure  # wichtig: immer ein gültiges Figure-Objekt zurückgeben

    x = np.arange(len(shifts))
    ax.plot(x, tard, marker="o", color="brown", label=tardiness_label)
    if earl is not None:
        ax.plot(x, earl, marker="o", color="orange", label=earliness_label)
    ax.set_xticks(x, shifts)
    return fig, ax




# Module call -------------------------------------------------------------------------------------------------------
def make_fig_tardiness_earliness_only(
    df_metrics: pd.DataFrame,
    experiment_id: int | None = None,
    shift_column: str = "Shift",
    tardiness_column: str = "Tardiness",
    earliness_column: str = "Earliness",
    experiment_column: str = "Experiment_ID",
    y_max: float | None = None,
    y_step: int = 60,
    show_earliness: bool = True,
    title: str = "Durchschnittliche Verspätung & Earliness je Schicht",
    tardiness_label: str = "Durchschnittliche Verspätung",
    earliness_label: str = "Durchschnittliche Earliness",
):
    """
    Convenience-Wrapper: erzeugt eine Linien-Figur (Ø Tardiness/Earliness je Schicht)
    für EIN Experiment aus df_metrics. Gibt fig zurück.
    Erwartet df_metrics mit Spalten [Shift, Tardiness, Earliness] und optional Experiment_ID.
    """
    if df_metrics.empty:
        raise ValueError("df_metrics ist leer.")

    # Falls Experiment-Spalte existiert: auf gewünschte ID filtern (oder erste nehmen)
    if experiment_column in df_metrics.columns:
        exps = df_metrics[experiment_column].dropna().unique()
        if len(exps) == 0:
            raise ValueError(f"Keine Werte in Spalte '{experiment_column}' gefunden.")
        exp_used = experiment_id if experiment_id is not None else exps[0]
        df_m = df_metrics.loc[df_metrics[experiment_column] == exp_used].copy()
        if df_m.empty:
            raise ValueError(f"Kein Eintrag für {experiment_column}={exp_used} in df_metrics.")
    else:
        exp_used = experiment_id
        df_m = df_metrics.copy()

    # Aggregation je Schicht
    df_shift = (
        df_m.groupby(shift_column, as_index=False)
            .agg(
                avg_tardiness=(tardiness_column, "mean"),
                avg_earliness=(earliness_column, "mean"),
            )
            .sort_values(shift_column)
    )

    # y-Maximum bestimmen (+15 Puffer, Untergrenze -15)
    if show_earliness:
        data_max = float(max(df_shift["avg_tardiness"].max(), df_shift["avg_earliness"].max()))
    else:
        data_max = float(df_shift["avg_tardiness"].max())
    y_top = (y_max if y_max is not None else data_max) + 15
    if y_top <= 0:
        y_top = 15

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_shift[shift_column], df_shift["avg_tardiness"], marker="o",
            color="brown", label=tardiness_label)
    if show_earliness:
        ax.plot(df_shift[shift_column], df_shift["avg_earliness"], marker="o",
                color="orange", label=earliness_label)

    ax.set_xlabel("Schicht (Shift)")
    ax.set_ylabel("Zeitabweichung")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_ylim(-15, y_top)
    ax.axhline(0, linewidth=0.8, color="black")
    ax.set_yticks(np.arange(0, y_top + 1, y_step))
    fig.tight_layout()
    return fig
