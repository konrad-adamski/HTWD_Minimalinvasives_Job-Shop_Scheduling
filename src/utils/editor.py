import pandas as pd


def enrich_schedule_dframe(df_schedule, df_jobs_information, on="Job"):
    """
    Enrich a schedule DataFrame with additional job information using a right join.

    This function performs a right join on the specified key (default: "Job") and automatically
    removes any duplicate columns from the job info DataFrame to avoid conflicts.
    Shared columns from the schedule are preserved; job info columns are prioritized.

    :param df_schedule: The schedule DataFrame (left side of the join).
    :param df_jobs_information: The job information DataFrame (right side of the join).
    :param on: The column name to join on.
    :return: A merged DataFrame containing all rows from `df_schedule` with enriched job info.
    """
    common_cols = set(df_schedule.columns) & set(df_jobs_information.columns) - {on}
    df_jobs_clean = df_jobs_information.drop(columns=list(common_cols))
    df_merged = pd.merge(df_schedule, df_jobs_clean, on=on, how="left")
    return df_merged