import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plotfig_empirical_flow_budget_distributions(
                df_times: pd.DataFrame, deadline_column = 'Deadline',
                bins: int = 30, y_max: float = 0.001):
    """
    Plot flow budget histograms for each routing group.

    For each group in the ``Routing_ID`` column, this function plots a histogram
    of flow budgets, defined as the difference between a target time (e.g. deadline)
    and the actual start of production. A vertical red line shows the average
    realized time usage in the group.

    All subplots share the same x- and y-axis limits for comparability.

    :param df_times: DataFrame containing routing and timing information.
                     Must include the columns ``Routing_ID``, ``End``,
                     ``Start of Production`` (or equivalent), and the column
                     specified via ``deadline_column``.
    :type df_times: pandas.DataFrame
    :param deadline_column: Name of the column representing the target or deadline time.
                            Default is ``'Deadline'``.
    :type deadline_column: str
    :param bins: Number of bins to use for the histogram. Default is 30.
    :type bins: int
    :param y_max: Maximum value of the y-axis (density). Default is 0.001.
    :type y_max: float

    :return: A matplotlib Figure object containing the subplots.
    :rtype: matplotlib.figure.Figure
    """
    groups = df_times['Routing_ID'].unique()
    n_groups = len(groups)
    n_cols = min(4, n_groups)
    n_rows = int(np.ceil(n_groups / n_cols))

    # global x-axis
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

    for j in range(idx + 1, n_rows * n_cols):
        fig.delaxes(axes[j // n_cols][j % n_cols])

    plt.tight_layout()
    return fig
