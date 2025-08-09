import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes


def get_scheduling_window_density_plot_figure(
        df_times: pd.DataFrame, routing_column: str = "Routing_ID", simulated_end_column: str = "End",
        earliest_start_column: str = "Ready Time", due_date_column: str = "Due Date",
        bins: int = 30, y_max: float = 0.015):
    """
    Create a density plot figure of scheduling windows for each routing.

    A scheduling window is defined as the time between the earliest start and the due date.
    A vertical red line shows the average (simulated) elapsed time from the earliest start to simulated end.
    A green line shows the average scheduling window for that group.

    All subplots share the same x- and y-axis limits for comparability.

    :param df_times: DataFrame with routing and timing information.
    :param routing_column: Column with routing group identifiers.
    :param simulated_end_column: Column with simulated end times.
    :param earliest_start_column: Column with the earliest possible start times.
    :param due_date_column: Column with due dates.
    :param bins: Number of histogram bins.
    :param y_max: Max value of the y-axis density.
    :return: Matplotlib Figure with the subplots.
    """
    routings = df_times[routing_column].unique()
    n_routings = len(routings)
    n_cols = min(4, n_routings)
    n_rows = int(np.ceil(n_routings / n_cols))

    # Global x-axis range
    all_scheduling_windows = df_times[due_date_column] - df_times[earliest_start_column]
    x_min, x_max = all_scheduling_windows.min(), all_scheduling_windows.max()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.ravel()

    for idx, routing in enumerate(routings):
        ax: Axes = axes[idx]
        dfr = df_times[df_times[routing_column] == routing]

        # Scheduling window = due date - earliest start
        scheduling_windows = dfr[due_date_column] - dfr[earliest_start_column]
        avg_scheduling_window = scheduling_windows.mean()

        # Average actual elapsed time = simulated end - earliest start
        avg_elapsed_time = (dfr[simulated_end_column] - dfr[earliest_start_column]).mean()

        sns.histplot(
            scheduling_windows, bins=bins, kde=True, stat="density",
            ax=ax, color="cornflowerblue", edgecolor="black"
        )
        ax.axvline(
            avg_elapsed_time,
            color='red',
            linestyle='--',
            label="Avg. elapsed time (simulated end - earliest start)"
        )
        ax.axvline(
            avg_scheduling_window,
            color="green",
            linestyle="--",
            label="Avg. scheduling window (due date - earliest start)"
        )

        ax.set_title(f'Routing {routing}')
        ax.set_xlabel("Time span from earliest start")
        ax.set_ylabel('Density')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)
        ax.legend()

    # Remove unused subplot axes
    for j in range(n_routings, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig
