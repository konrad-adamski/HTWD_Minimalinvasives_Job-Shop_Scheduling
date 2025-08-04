from typing import Optional

import pandas as pd
import numpy as np


def add_groupwise_lognormal_deadlines_by_group_mean(
        df_times_temp: pd.DataFrame, sigma: float = 0.2, routing_column: str = "Routing_ID",
        earliest_start_column ="Ready Time", end_column: str = "End", deadline_column: str = "Deadline",
        seed: Optional[int] = 42) -> pd.DataFrame:
    """
    Generate stochastic deadlines per routing group using log-normal-distributed flow budgets.

    For each group in the specified routing column, a log-normal distribution is fitted such that
    its mean matches the group's average flow time (End - Ready Time). Each deadline is then sampled
    individually per job and added to the respective 'Ready Time'.

    :param df_times_temp: DataFrame containing at least the columns 'Ready Time', 'End', and the routing group column.
    :type df_times_temp: pandas.DataFrame
    :param sigma: Standard deviation of the log-normal distribution in log-space (default: 0.2).
    :type sigma: float
    :param routing_column: Column used to group jobs for separate deadline distributions (default: "Routing_ID").
    :type routing_column: str
    :param seed: Random seed for reproducibility (default: 42).
    :type seed: int
    :return: Copy of the input DataFrame with an additional column 'Deadline'.
    :rtype: pandas.DataFrame
    """
    if seed is not None:
        np.random.seed(seed)

    df_times = df_times_temp.copy()
    df_times[deadline_column] = np.nan

    for routing_id, grp in df_times.groupby(routing_column):
        target_flow_mean = grp[end_column].mean() - grp[earliest_start_column].mean()
        mu = np.log(target_flow_mean) - 0.5 * sigma**2

        # Für jede Zeile in Gruppe eine Deadline aus LogNormal(mu, sigma)
        flow_budgets = np.random.lognormal(mean=mu, sigma=sigma, size=len(grp))
        df_times.loc[grp.index, deadline_column] = df_times.loc[grp.index, earliest_start_column] + np.round(flow_budgets)

    return df_times


def ensure_reasonable_deadlines(
        df_times: pd.DataFrame, min_coverage: float = 0.90,
        earliest_start_column = "Ready Time", end_column: str = "End",
        deadline_column: str = "Deadline", total_duration_column = "Total Processing Time") -> pd.DataFrame:
    """
    Ensures that each job's deadline covers at least a minimum percentage of its total processing time.
    Also removes the 'End' column if present.

    :param df_times: DataFrame with at least the columns ['Ready Time', 'Deadline', 'Job Processing Time'].
    :param min_coverage: Minimum fraction (0–1) of the job processing time that must be covered by the deadline.
                         Defaults to 0.90 (i.e. 90% of total processing time).
    :return: DataFrame with updated 'Deadline' values and 'End' column removed if present.
    """
    min_coverage = min(min_coverage, 1.0)

    df_times[deadline_column] = np.maximum(
        df_times[deadline_column],
        df_times[earliest_start_column] + df_times[total_duration_column] * min_coverage
    )

    df_times[deadline_column] = np.ceil(df_times[deadline_column]).astype(int)

    if end_column in df_times.columns:
        df_times = df_times.drop(columns=[end_column])

    return df_times