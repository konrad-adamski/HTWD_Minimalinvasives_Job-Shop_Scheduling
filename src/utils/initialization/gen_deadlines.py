from typing import Optional

import pandas as pd
import numpy as np

def get_temporary_df_times_from_schedule(
        df_schedule: pd.DataFrame,
        df_jssp: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Prepare job-level timing summary for routing-based deadline generation.

    This function extracts for each job its arrival, ready, and end time,
    as well as its total processing time (optional),
    to support deadline generation methods that group jobs by routing.

    :param df_schedule: Schedule DataFrame with columns
        'Job', 'Operation', 'Routing_ID', 'Arrival', 'Ready Time', and 'End'.
    :param df_jssp: Optional job-shop definition with 'Job' and 'Processing Time' columns.
        If provided, the total processing time per job is computed and included.
    :return: DataFrame with columns
        'Job', 'Routing_ID', 'Arrival', 'Ready Time', 'End'
        and optionally 'Job Processing Time' if df_jssp is provided.
    """
    # Select the last operation for each job
    df_last_ops = df_schedule.sort_values("Operation").groupby("Job").last().reset_index()

    # Base columns from the schedule
    df_jobs_times = df_last_ops[["Job", "Routing_ID", "Arrival", "Ready Time", "End"]]

    # Optionally add job-level total processing time
    if df_jssp is not None:
        df_proc_time = df_jssp.groupby("Job", as_index=False)["Processing Time"].sum()
        df_proc_time.rename(columns={"Processing Time": "Job Processing Time"}, inplace=True)
        df_jobs_times = df_jobs_times.merge(df_proc_time, on="Job", how="left")

    return df_jobs_times


def add_groupwise_lognormal_deadlines_by_group_mean(
                df_times_temp: pd.DataFrame, sigma: float = 0.2,
                routing_column: str = "Routing_ID", seed: Optional[int] = 42) -> pd.DataFrame:
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
    df_times['Deadline'] = np.nan

    for routing_id, grp in df_times.groupby(routing_column):
        target_flow_mean = grp['End'].mean() - grp['Ready Time'].mean()
        mu = np.log(target_flow_mean) - 0.5 * sigma**2

        # Für jede Zeile in Gruppe eine Deadline aus LogNormal(mu, sigma)
        flow_budgets = np.random.lognormal(mean=mu, sigma=sigma, size=len(grp))
        df_times.loc[grp.index, 'Deadline'] = df_times.loc[grp.index, 'Ready Time'] + np.round(flow_budgets)

    return df_times


def ensure_reasonable_deadlines(df_times: pd.DataFrame, min_coverage: float = 0.90) -> pd.DataFrame:
    """
    Ensures that each job's deadline covers at least a minimum percentage of its total processing time.
    Also removes the 'End' column if present.

    :param df_times: DataFrame with at least the columns ['Ready Time', 'Deadline', 'Job Processing Time'].
    :param min_coverage: Minimum fraction (0–1) of the job processing time that must be covered by the deadline.
                         Defaults to 0.90 (i.e. 90% of total processing time).
    :return: DataFrame with updated 'Deadline' values and 'End' column removed if present.
    """
    min_coverage = min(min_coverage, 1.0)

    df_times['Deadline'] = np.maximum(
        df_times['Deadline'],
        df_times['Ready Time'] + df_times['Job Processing Time'] * min_coverage
    )

    df_times['Deadline'] = np.ceil(df_times['Deadline']).astype(int)

    if 'End' in df_times.columns:
        df_times = df_times.drop(columns=['End'])

    return df_times