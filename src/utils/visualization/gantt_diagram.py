import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# GANTT Diagramme --------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------

# Farbskala: tab20 mit Überspringen von Index 6 und Layer-Anpassungen
tab20 = plt.get_cmap("tab20")

def _get_color(idx):
    base_idx = idx % 16
    layer = idx // 16
    # --- Anpassung: überspringe Index 6 ---
    if base_idx >= 6:
        base_idx += 1  # 6 wird übersprungen
    rgba = tab20(base_idx / 20)  # Skaliere auf 20 Farben
    r, g, b, _ = rgba
    if layer == 1:
        r = max(0.0, r * 0.9)
        g = min(1.0, g * 1.4)
        b = max(0.0, b * 0.9)
    elif layer == 2:
        r = min(1.0, r * 1.15)
        g = max(0.0, g * 0.85)
        b = min(1.0, b * 1.15)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


def get_plot(schedules_df: pd.DataFrame,
               title: str = "Gantt-Diagramm",
               job_id_column: str = "Job",
               duration_column: str = "Processing Time",
               perspective: str = "Job"):
    """
    Zeichnet ein Gantt-Diagramm aus Sicht von Schedules (Aufträgen) oder Maschinen.

    Parameters:
    - schedules_df: DataFrame mit den Operationsdaten
    - title: Titel des Diagramms
    - job_id_column: Spalte mit Auftrags-ID (nur relevant für "Job"-Perspektive)
    - duration_column: Spalte mit Bearbeitungszeit
    - perspective: "Job" (Auftragsperspektive) oder "Machine" (Maschinenperspektive)
    """

    if perspective not in ["Job", "Machine"]:
        raise ValueError("perspective muss 'Job' oder 'Machine' sein")

    # Achsen- und Farbsteuerung
    if perspective == "Job":
        group_column = job_id_column
        color_column = "Machine"
        ylabel = "Produktionsaufträge"
    else:  # perspective == "Machine"
        group_column = "Machine"
        color_column = job_id_column
        ylabel = "Maschinen"

    groups = sorted(schedules_df[group_column].unique())
    color_items = sorted(schedules_df[color_column].unique())
    yticks = range(len(groups))

    color_map = {item: _get_color(i) for i, item in enumerate(color_items)}

    fig_height = len(groups) * 0.8
    fig, ax = plt.subplots(figsize=(16, fig_height))

    for idx, group in enumerate(groups):
        rows = schedules_df[schedules_df[group_column] == group]
        for _, row in rows.iterrows():
            ax.barh(idx,
                    row[duration_column],
                    left=row['Start'],
                    height=0.5,
                    color=color_map[row[color_column]],
                    edgecolor='black')

    # Legende
    legend_handles = [mpatches.Patch(color=color_map[item], label=str(item)) for item in color_items]
    legend_columns = (len(color_items) // 35) + 1
    ax.legend(handles=legend_handles,
              title=color_column,
              bbox_to_anchor=(1.01, 1),
              loc='upper left',
              ncol=legend_columns,
              handlelength=2.4,
              frameon=False,  # Kein Rahmen
              alignment='left'
              )

    ax.set_yticks(yticks)
    ax.set_yticklabels(groups)
    ax.set_xlabel("Zeit (in Minuten)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Achsenlimits
    max_time = (schedules_df['Start'] + schedules_df[duration_column]).max()
    x_start = int((schedules_df['Start'].min() // 1440) * 1440)
    ax.set_xlim(x_start, max_time + 60)

    xticks = list(range(x_start, int(max_time) + 360, 360))
    ax.set_xticks(xticks)
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

    for x in range(x_start, int(max_time) + 1440, 1440):
        ax.axvline(x=x, color='#777777', linestyle='-', linewidth=1.0, alpha=0.6)

    plt.tight_layout()
    plt.show()
