from typing import Optional, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.analyses.fig_startdeviation import prepare_start_deviation_series, draw_start_deviation_bars
from src.analyses.fig_tardiness_earliness import prepare_avg_metrics_series, draw_avg_metrics_lines


def make_combined_figure_startdeviation_tardiness_earliness(
    df_dev: pd.DataFrame,
    df_metrics: pd.DataFrame,
    experiment_id: Optional[int] = None,
    shift_column: str = "Shift",
    deviation_column: str = "Deviation",
    experiment_column: str = "Experiment_ID",
    tardiness_column: str = "Tardiness",
    earliness_column: str = "Earliness",
    show_earliness: bool = True,
    y_max: Optional[float] = None,
    y_step: int = 60,
    title: str = "Zeitabweichungen je Schicht",
    bar_label: str = "Ø Startdeviation",
    tardiness_label: str = "Ø Tardiness",
    earliness_label: str = "Ø Earliness",
) -> plt.Figure:
    """
    Nutzt die beiden Module, richtet die Shifts aus und zeichnet Balken + Linien auf eine Achse.
    Gibt die Figure zurück.
    """
    # Serien vorbereiten (je Modul)
    dev_spec = prepare_start_deviation_series(
        df_dev, experiment_id, shift_column, deviation_column, experiment_column
    )
    met_spec = prepare_avg_metrics_series(
        df_metrics, experiment_id, shift_column, tardiness_column, earliness_column, experiment_column, show_earliness
    )

    # Gemeinsame x-Achse (Shifts vereinigen & neu ausrichten)
    shifts_all = sorted(set(dev_spec["shifts"]) | set(met_spec["shifts"]))

    def _align(series_shifts: List, values: np.ndarray, fill: float) -> np.ndarray:
        pos = {s: i for i, s in enumerate(series_shifts)}
        return np.array([values[pos[s]] if s in pos else fill for s in shifts_all], dtype=float)

    bar_vals  = _align(dev_spec["shifts"], dev_spec["values"], 0.0)
    line_tard = _align(met_spec["shifts"], met_spec["tard"],  np.nan)
    line_earl = _align(met_spec["shifts"], met_spec["earl"],  np.nan) if met_spec["earl"] is not None else None

    # y_max bestimmen (+15 Puffer, Untergrenze -15)
    candidates = [np.nanmax(bar_vals), np.nanmax(line_tard)]
    if line_earl is not None:
        candidates.append(np.nanmax(line_earl))
    data_max = float(np.nanmax(candidates)) if candidates else 0.0
    y_top = (y_max if y_max is not None else data_max) + 15
    if y_top <= 0:
        y_top = 15

    # Zeichnen (eine Figure, ein Axes)
    fig, ax = plt.subplots(figsize=(12, 6))
    _, ax = draw_start_deviation_bars(shifts_all, bar_vals, ax, bar_label)
    _, ax = draw_avg_metrics_lines(shifts_all, line_tard, line_earl, ax, tardiness_label, earliness_label)

    # Achsen & Deko
    ax.set_xlabel("Schicht (Shift)")
    ax.set_ylabel("Zeitabweichung")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(-15, y_top)
    ax.axhline(0, linewidth=0.8, color="black")
    ax.set_yticks(np.arange(0, y_top + 1, y_step))
    ax.legend()
    fig.tight_layout()
    return fig