from typing import Optional

import numpy as np
import pandas as pd


class DataFrameEnrichment:
    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated.")

    # Deadlines ------------------------------------------------------------------------------------------------------
    @staticmethod
    def add_groupwise_lognormal_deadlines_by_group_mean(
            df_times_temp: pd.DataFrame, sigma: float = 0.2, routing_column: str = "Routing_ID",
            earliest_start_column="Ready Time", end_column: str = "End", deadline_column: str = "Deadline",
            seed: Optional[int] = 42) -> pd.DataFrame:
        """
        Generate stochastic deadlines per routing group using log-normal-distributed flow budgets.

        For each group in the specified routing column, a log-normal distribution is fitted such that
        its mean matches the group's average flow time (End - Ready Time). Each deadline is then sampled
        individually per job and added to the respective 'Ready Time'.

        :param df_times_temp: DataFrame containing at least the columns 'Ready Time', 'End', and the routing group column.
        :param sigma: Standard deviation of the log-normal distribution in log-space.
        :param routing_column: Column used to group jobs for separate deadline distributions.
        :param seed: Random seed for reproducibility.
        :return: Copy of the input DataFrame with an additional column 'Deadline'.
        """
        if seed is not None:
            np.random.seed(seed)

        df_times = df_times_temp.copy()
        df_times[deadline_column] = np.nan

        for routing_id, grp in df_times.groupby(routing_column):
            target_flow_mean = grp[end_column].mean() - grp[earliest_start_column].mean()
            mu = np.log(target_flow_mean) - 0.5 * sigma ** 2

            # Für jede Zeile in Gruppe eine Deadline aus LogNormal(mu, sigma)
            flow_budgets = np.random.lognormal(mean=mu, sigma=sigma, size=len(grp))
            df_times.loc[grp.index, deadline_column] = df_times.loc[grp.index, earliest_start_column] + np.round(
                flow_budgets)

        return df_times

    @staticmethod
    def ensure_reasonable_deadlines(
            df_times: pd.DataFrame, min_coverage: float = 0.90,
            earliest_start_column="Ready Time", end_column: str = "End",
            deadline_column: str = "Deadline", total_duration_column="Total Processing Time") -> pd.DataFrame:
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


    # Transition times -----------------------------------------------------------------------------------------------
    @staticmethod
    def _compute_transition_times_per_job_backward(
            df_schedule: pd.DataFrame, job_column: str = "Job", position_number_column: str = "Operation",
            start_column: str = "Start", end_column: str = "End", new_transition_column: str = "Transition Time"):
        """
        Vectorized backward scheduling transition times.

        :param df_schedule: Schedule DataFrame.
        :param job_column: Job ID column name.
        :param position_number_column: Operation sequence column name.
        :param start_column: Start time column name.
        :param end_column: End time column name.
        :param new_transition_column: Output transition time column name.
        :return: DataFrame with added transition time column.
        """
        df = df_schedule.copy()
        df = df.sort_values([job_column, position_number_column], ascending=[True, False])
        df['End_Previous_Operation'] = df.groupby(job_column)[end_column].shift(-1)
        df[new_transition_column] = df[start_column] - df['End_Previous_Operation']
        return df

    @staticmethod
    def _compute_avg_transition_times_per_machine(
            df_jobs_transition_times: pd.DataFrame, machine_column: str = "Machine",
            transition_column: str = "Transition Time",
            new_avg_transition_column: str = "Ø Transition_Time") -> pd.DataFrame:
        """
        Compute average transition time per machine.

        :param df_jobs_transition_times: DataFrame with machine and transition time columns.
        :param machine_column: Name of the machine column.
        :param transition_column: Name of the transition time column.
        :param new_avg_transition_column: Name for the resulting average transition time column.
        :return: DataFrame with columns ``machine_column`` and ``new_avg_transition_column``.
        :raises ValueError: If required columns are missing.
        """
        if not {machine_column, transition_column}.issubset(df_jobs_transition_times.columns):
            raise ValueError(f"DataFrame must contain '{machine_column}' and '{transition_column}' columns.")

        return (df_jobs_transition_times.groupby(machine_column)[transition_column]
            .mean()
            .round(0)
            .astype(int)
            .reset_index()
            .rename(columns={transition_column: new_avg_transition_column})
        )

    @classmethod
    def compute_avg_transition_times_per_machine_backward(
            cls, df_schedule: pd.DataFrame, job_column: str = "Job", machine_column: str = "Machine",
            position_number_column: str = "Operation", start_column: str = "Start", end_column: str = "End",
            new_avg_transition_column: str = "Ø Transition Time"):
        """
         Compute average transition times per machine for backward scheduling.

         Transition times are first calculated per job, then averaged per machine.

         :param df_schedule: Schedule DataFrame.
         :param job_column: Job ID column name.
         :param machine_column: Machine column name.
         :param position_number_column: Operation sequence column name.
         :param start_column: Start time column name.
         :param end_column: End time column name.
         :param new_avg_transition_column: Output column name for average transition time.
         :return: DataFrame with average transition time per machine.
         """
        df_jobs_transition_times = cls._compute_transition_times_per_job_backward(
            df_schedule = df_schedule,
            job_column=job_column,
            position_number_column=position_number_column,
            start_column=start_column,
            end_column=end_column,
        )
        return cls._compute_avg_transition_times_per_machine(
            df_jobs_transition_times = df_jobs_transition_times,
            machine_column= machine_column,
            new_avg_transition_column= new_avg_transition_column
        )

