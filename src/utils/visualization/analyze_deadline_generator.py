import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plotfig_empirical_flow_budget_distributions(df_times: pd.DataFrame, deadline_column = 'Deadline', bins: int = 30, y_max: float = 0.001):
    """
    Plottet für jede Gruppe in 'Routing_ID' das Histogramm + KDE
    der Flow Budgets (Deadline - Ready Time) auf Basis von df_times.
    Zusätzlich wird der Zielwert (End.mean - Ready Time.mean) als rote Linie eingezeichnet.
    Alle Subplots verwenden dieselbe X- und Y-Achse.

    Parameters
    ----------
    df_times : pd.DataFrame
        Muss Spalten 'Routing_ID', 'Deadline', 'Ready Time', 'End' enthalten.
    bins : int
        Anzahl der Bins im Histogramm.
    y_max : float
        Maximaler Y-Wert (Dichte) für die Achsenskalierung. Default ist 0.001.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Die erzeugte Figure mit den Subplots.
    """
    groups = df_times['Routing_ID'].unique()
    n_groups = len(groups)
    n_cols = min(4, n_groups)
    n_rows = int(np.ceil(n_groups / n_cols))

    # Globale X-Achse basierend auf Flow Budgets
    all_flow_budgets = df_times[deadline_column] - df_times['Ready Time']
    x_min, x_max = all_flow_budgets.min(), all_flow_budgets.max()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, group in enumerate(groups):
        ax = axes[idx // n_cols][idx % n_cols]
        grp = df_times[df_times['Routing_ID'] == group]
        flow_budgets = grp[deadline_column] - grp['Ready Time']
        target = grp['End'].mean() - grp['Ready Time'].mean()

        sns.histplot(flow_budgets, bins=bins, kde=True, stat='density',
                     ax=ax, color='cornflowerblue', edgecolor='black')

        ax.axvline(target, color='red', linestyle='--', label='Target')
        ax.set_title(f'Arbeitsplan {group}')
        ax.set_xlabel(f'Flow Budget ({deadline_column} - Ready Time)')
        ax.set_ylabel('Dichte')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)
        ax.legend()

    # Leere Plots deaktivieren
    for j in range(idx + 1, n_rows * n_cols):
        fig.delaxes(axes[j // n_cols][j % n_cols])

    plt.tight_layout()
    return fig
