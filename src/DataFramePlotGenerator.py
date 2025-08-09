import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from typing import Literal
from matplotlib.axes import Axes
from matplotlib import pyplot as plt


class DataFramePlotGenerator:
    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated.")

    # Define base colormap
    tab20 = plt.get_cmap("tab20")

    @classmethod
    def _get_color(cls, idx):
        """
        Generate a distinct color from the tab20 colormap with index correction
        and layer-based variation to extend the palette.
        - Adjusts RGB values for every 16-color cycle to create new color shades.
        :param idx: Integer index of the item
        :return: Hex color code as string
        """
        base_idx = idx % 16
        layer = idx // 16
        # --- Adjustment: skip index 6 ---
        if base_idx >= 6:
            base_idx += 1

        # Scale to 20 colors
        rgba = cls.tab20(base_idx / 20)
        r, g, b, _ = rgba
        if layer == 1:
            r, g, b = max(0.0, r * 0.9), min(1.0, g * 1.4), max(0.0, b * 0.9)
        elif layer == 2:
            r, g, b = min(1.0, r * 1.15), max(0.0, g * 0.85), min(1.0, b * 1.15)

        return f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'

    # Check due date ------------------------------------------------------------------------------------------------
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
            ax: Axes = axes[idx]    # type: ignore
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
            fig.delaxes(axes[j])    # type: ignore

        plt.tight_layout()
        return fig

    # Gantt chart for schedule or simulation -------------------------------------------------------------------------
    @classmethod
    def get_gantt_chart_figure(
            cls, df_workflow: pd.DataFrame, title: str = "Gantt chart",
            job_column: str = "Job", machine_column: str = "Machine", duration_column: str = "Processing Time",
            perspective: Literal["Machine", "Job"] = "Machine"):
        """
        Create a Gantt chart figure from either a job or machine perspective.

        :param df_workflow: DataFrame containing scheduling or simulation data
        :param title: Title of the chart
        :param job_column: Column name identifying jobs
        :param machine_column: Column name identifying machines
        :param duration_column: Column name for operation durations
        :param perspective: Either "Job" (job-centric view) or "Machine" (machine-centric view)
        :return: Matplotlib Figure object
        """

        # Axis and color settings
        if perspective == "Job":
            group_column = job_column
            color_column = machine_column
        elif  perspective == "Machine":
            group_column = machine_column
            color_column = job_column
        else:
            raise ValueError("Perspective must be 'Job' or 'Machine'")
        y_label = group_column

        groups = sorted(df_workflow[group_column].unique())
        color_items = sorted(df_workflow[color_column].unique())
        y_ticks = range(len(groups))
        color_map = {item: cls._get_color(i) for i, item in enumerate(color_items)}

        fig_height = len(groups) * 0.8
        fig, ax = plt.subplots(figsize=(16, fig_height))

        for idx, group in enumerate(groups):
            rows = df_workflow[df_workflow[group_column] == group]
            for _, row in rows.iterrows():
                ax.barh(idx,
                        row[duration_column],
                        left=row['Start'],
                        height=0.5,
                        color=color_map[row[color_column]],
                        edgecolor='black')

        # Legend
        legend_handles = [mpatches.Patch(color=color_map[item], label=str(item)) for item in color_items]
        legend_columns = (len(color_items) // 35) + 1
        ax.legend(handles=legend_handles,
                  title=color_column,
                  bbox_to_anchor=(1.01, 1),
                  loc='upper left',
                  ncol=legend_columns,
                  handlelength=2.4,
                  frameon=False,
                  alignment='left'
                  )

        # Axis labels and formatting
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(groups)
        ax.set_xlabel("Time (in minutes)")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        # Time axis scaling
        max_time = (df_workflow['Start'] + df_workflow[duration_column]).max()
        x_start = int((df_workflow['Start'].min() // 1440) * 1440)
        ax.set_xlim(x_start, max_time + 60)

        x_ticks = list(range(x_start, int(max_time) + 360, 360))
        ax.set_xticks(x_ticks)
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)

        # Vertical lines every 1440 minutes (e.g., day delimiter)
        for x in range(x_start, int(max_time) + 1440, 1440):
            ax.axvline(x=x, color='#777777', linestyle='-', linewidth=1.0, alpha=0.7)

        plt.tight_layout()
        return fig

