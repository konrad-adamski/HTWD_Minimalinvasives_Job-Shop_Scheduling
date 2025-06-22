import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def plot_relative_spread_horizontal(df, end_col: str = 'End', deadline_col: str = 'Deadline', figsize: tuple = (12, 3), 
                                    title: str = 'Range-Boxplot der relativen Streuung') -> plt.Figure:
    """
    Zeichnet einen horizontalen Range-Boxplot der relativen Streuung
    und gibt die Figure zurück.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Die erzeugte Figure.
    """
    # Relative Streuung berechnen
    rel = (df[deadline_col] - df[end_col]) / df[end_col]
    fig, ax = plt.subplots(figsize=figsize)

    # Horizontaler Range-Boxplot
    bp = ax.boxplot(
        rel,
        vert=False,
        patch_artist=True,
        showcaps=True,
        whis=(0, 100),
        widths=0.4,
        boxprops=dict(facecolor='#ededed', edgecolor='darkslategray'),
        medianprops=dict(color='#ff3300', linewidth=2),
        whiskerprops=dict(color='darkslategray'),
        capprops=dict(color='darkslategray'),
        showfliers=False
    )

    # Echte Whisker-Enden auslesen und Achsen-Limits setzen
    lower = bp['whiskers'][0].get_xdata()[1]
    upper = bp['whiskers'][1].get_xdata()[1]
    ax.set_xlim(lower, upper)

    # X-Ticks in 0.1-Schritten
    start = np.floor(lower * 10) / 10
    stop  = np.ceil(upper * 10) / 10
    xticks = np.arange(start, stop + 0.1, 0.1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.1f}" for x in xticks])

    # Styling
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.set_xlabel('Relative Streuung\n(Deadline - End) / End')
    ax.set_title(title)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig



def plot_relative_spread_histograms(
    df_times: pd.DataFrame,
    plan_col: str = 'Production_Plan_ID',
    deadline_col: str = 'Deadline',
    end_col: str = 'End',
    compute_global_bounds: bool = True,
    bounds_padding: float = 0.5,
    n_bins: int = 80,
    figsize: tuple = (10, 4),
    max_cols: int = 2,
    show_grid: bool = True
) -> plt.Figure:
    """
    Zeichnet für jede Produktionsplan-ID ein Dichte-Histogramm der relativen Streuung
    auf Subplots in einer einzigen Figure und gibt diese Figure zurück.
    Legenden werden rechts außerhalb der einzelnen Subplots platziert.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Die erzeugte Figure mit allen Subplots.
    """
    # 1. Relative Streuung berechnen
    df = df_times.copy()
    df['rel'] = (df[deadline_col] - df[end_col]) / df[end_col]

    # 2. Globale Grenzen (optional)
    if compute_global_bounds:
        global_min = df['rel'].min() - bounds_padding
        global_max = df['rel'].max() + bounds_padding
        bins = np.linspace(global_min, global_max, n_bins + 1)
    else:
        bins = n_bins

    # 3. Subplot-Grid vorbereiten
    plan_ids = sorted(df[plan_col].unique())
    n = len(plan_ids)
    n_cols = min(max_cols, n)
    n_rows = math.ceil(n / n_cols)

    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
        squeeze=False
    )
    axs = axs.flatten()

    # Raum an der rechten Seite freigeben für Legenden
    fig.subplots_adjust(right=0.8)

    # 4. Für jede Plan-ID ein Histogramm in das jeweilige Subplot
    for idx, plan_id in enumerate(plan_ids):
        ax = axs[idx]
        rel = df.loc[df[plan_col] == plan_id, 'rel']
        rel_min, rel_max = rel.min(), rel.max()

        ax.hist(rel, bins=bins, density=True, edgecolor='white', alpha=0.8)
        if compute_global_bounds:
            ax.set_xlim(global_min, global_max)

        # Linien mit Label für die Legende
        ax.axvline(rel_min, linestyle='--', linewidth=1.5,
                   alpha=0.6, color='orange', label=f'Min ({rel_min:.2f})')
        ax.axvline(rel_max, linestyle='--', linewidth=1.5,
                   alpha=0.6, color='red',    label=f'Max ({rel_max:.2f})')

        ax.set_title(f'Produktionsplan {plan_id}')
        ax.set_xlabel('Relative Streuung')
        ax.set_ylabel('Dichte')
        if show_grid:
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        # Legende außerhalb rechts
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    # 5. Leere Subplots ausblenden
    for j in range(n, n_rows * n_cols):
        fig.delaxes(axs[j])

    fig.tight_layout()
    return fig
