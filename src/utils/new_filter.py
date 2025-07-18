import pandas as pd

def get_current_jobs(df_jobs_times: pd.DataFrame, df_previous_not_started: pd.DataFrame | None = None,
                       ready_time: int = 0, ready_time_col = "Ready Time") -> pd.DataFrame:
    """
    Filtert Produktionsauftragsinformationen (Zeit-Informationen) nach aktuelle 'Ready Time'
    und nach den Produktionsauftragsaufträgen, die nicht begonnene Arbeitsgänge haben
    """
    # Aktuelle und unerledigte Jobs
    if df_previous_not_started is None or df_previous_not_started.empty:
        this_filter = df_jobs_times[ready_time_col] == ready_time
    else:
        this_filter = (df_jobs_times["Ready Time"] == ready_time) | (
            df_jobs_times["Job"].isin(df_previous_not_started["Job"].unique()))
    return df_jobs_times[this_filter]


def filter_current_jssp(df_jssp: pd.DataFrame, df_jobs_times_current: pd.DataFrame,
                                  exclusion_dataframes_list: list | None = None, job_column: str = "Job") -> pd.DataFrame:
    """
    Filtert aus df_jssp alle Job-Operationen heraus, die:
    - zu Jobs gehören, die heute angekommen sind (aus df_jobs_times_current),
    - und weder aktiv noch bereits ausgeführt wurden (aus exclusion_dataframes).
    """
    # 1. Jobs, die heute angekommen sind
    jobs_today = df_jobs_times_current[job_column].unique()

    # 2. JobOperation-Tupel, die ausgeschlossen werden sollen
    if exclusion_dataframes_list:
        excluded_job_operations = set(
            tuple(x) for x in pd.concat(exclusion_dataframes_list)[["Job", "Operation"]].to_numpy()
        )
    else:
        excluded_job_operations = set()

    # 3. Filter anwenden auf df_jssp
    df_jssp = df_jssp.copy()
    df_jssp["Job_Op_Tuple"] = list(zip(df_jssp[job_column], df_jssp["Operation"]))
    this_filter = df_jssp[job_column].isin(jobs_today) & ~df_jssp["Job_Op_Tuple"].isin(excluded_job_operations)

    return df_jssp[this_filter].drop(columns=["Job_Op_Tuple"])