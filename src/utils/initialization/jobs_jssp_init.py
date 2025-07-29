import pandas as pd
import numpy as np
import random

from typing import Optional
from src.utils.initialization.arrivals_init import calculate_mean_interarrival_time, \
    generate_arrivals_from_mean_interarrival_time


def create_jobs_for_shifts(
        df_routings: pd.DataFrame, routing_column: str = 'Routing_ID', job_column: str = 'Job',
        operation_column: str = 'Operation', machine_column: str = "Machine", duration_column: str = "Processing Time",
        arrival_column: str = "Arrival", ready_time_column: str = "Ready Time", shift_count: int = 1,
        shift_length: int = 1440, u_b_mmax: float = 0.9, shuffle: bool = False, job_seed: int = 50,
        arrival_seed: Optional[int] = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate jobs across multiple shifts with arrival times based on target machine utilization.

    This function expands routing templates into multiple jobs, distributes arrivals
    to match a utilization target, and assigns ready times aligned to shifts.

    :param df_routings: Routing template with columns [routing_column, operation_column, machine_column, duration_column]
    :param routing_column: Column name for routing identifiers
    :param job_column: Column name for generated job IDs
    :param operation_column: Column name for operation sequence
    :param machine_column: Column name for machine assignment
    :param duration_column: Column name for processing time
    :param arrival_column: Column name for job arrival times
    :param ready_time_column: Column name for earliest ready time per job
    :param shift_count: Number of production shifts
    :param shift_length: Length of a shift in minutes
    :param u_b_mmax: Target utilization of the bottleneck machine
    :param arrival_seed: Optional seed for arrival time generation
    :param job_seed: Random seed for job ID generation
    :param arrival_seed: Random seed for arrival time generation
    :return: Tuple of two DataFrames: (df_jssp with operations, df_arrivals with arrival and ready times)
    """

    # 1) Generate jobs using routing templates
    multiplication = 2 + shift_length // 500
    repetitions = multiplication * shift_count
    df_jssp = generate_multiple_jssp_from_routings(
        df_routings=df_routings,
        job_column=job_column,
        routing_column=routing_column,
        operation_column=operation_column,
        machine_column=machine_column,
        duration_column=duration_column,
        repetitions=repetitions,
        shuffle=shuffle, seed=job_seed
    )

    # 2) Compute mean interarrival time based on bottleneck utilization
    t_a = calculate_mean_interarrival_time(
        df_routings,
        u_b_mmax = u_b_mmax,
        routing_column= routing_column,
        machine_column= machine_column,
        duration_column= duration_column,
    )

    # 3) Generate randomized arrival times for all jobs
    unique_jobs = df_jssp[[job_column, routing_column]].drop_duplicates()
    job_numb = len(unique_jobs)
    arrivals = generate_arrivals_from_mean_interarrival_time(
        job_number=job_numb,
        mean_interarrival_time=t_a,
        var_type="Integer",
        random_seed=arrival_seed
    )

    df_jobs_arrivals = unique_jobs.copy()
    df_jobs_arrivals[arrival_column] = arrivals

    # 4a) Keep only jobs arriving within the planning window
    time_limit = shift_count * shift_length
    df_jobs_arrivals = df_jobs_arrivals[df_jobs_arrivals[arrival_column] < time_limit].reset_index(drop=True)

    # 4b) Keep corresponding jssp only
    valid_ids = set(df_jobs_arrivals[job_column])
    df_jssp = df_jssp[df_jssp[job_column].isin(valid_ids)].reset_index(drop=True)

    # 5) Compute ready times aligned to the next full shift
    df_jobs_arrivals = get_df_with_ready_time_by_shift(
        df_arrivals=df_jobs_arrivals,
        shift_length=shift_length,
        arrival_column=arrival_column,
        ready_time_column=ready_time_column
    )

    return df_jssp, df_jobs_arrivals
    

def generate_multiple_jssp_from_routings(
        df_routings: pd.DataFrame, job_column: str = 'Job', routing_column: str = 'Routing_ID',
        operation_column: str = 'Operation', machine_column: str = 'Machine', duration_column: str = 'Processing Time',
                                         repetitions: int = 3, shuffle: bool = False, seed: int = 50) -> pd.DataFrame:
    """
    Generate multiple sets of JSSP jobs by repeatedly calling `generate_jssp_from_routings`.

    Each repetition adds a new set of jobs based on the routing templates, using distinct job IDs.

    :param df_routings: DataFrame with routing definitions [routing_column, operation_column, machine_column, duration_column]
    :param job_column: Name of the job ID column
    :param routing_column: Name of the routing template ID column
    :param operation_column: Name of the operation index column
    :param machine_column: Name of the machine column
    :param duration_column: Name of the processing time column
    :param repetitions: Number of times to replicate the routing set
    :param shuffle: Whether to shuffle routing templates in each repetition
    :param seed: Base random seed (incremented per repetition)
    :return: Combined DataFrame with generated jobs
    """
    all_jobs = []
    routings_per_repetition = df_routings[routing_column].nunique()

    for i in range(repetitions):
        offset = i * routings_per_repetition
        current_seed = seed + i

        df_jobs = generate_jssp_from_routings(
            df_routings,
            job_column=job_column,
            routing_column=routing_column,
            operation_column= operation_column,
            machine_column= machine_column,
            duration_column= duration_column,
            offset=offset,
            shuffle=shuffle,
            seed=current_seed
        )
        all_jobs.append(df_jobs)

    return pd.concat(all_jobs, ignore_index=True)

def generate_jssp_from_routings(
        df_routings: pd.DataFrame, job_column: str = 'Job', routing_column: str = 'Routing_ID',
        operation_column: str = 'Operation', machine_column: str = 'Machine', duration_column: str = 'Processing Time',
        offset: int = 0, shuffle: bool = False, seed: int = 50) -> pd.DataFrame:
    """
       Generate new JSSP based on routing templates with sequential job IDs.

       Each job is created by copying a routing template (identified by `routing_column`).

       :param df_routings: DataFrame with routing definitions [routing_column, operation_column, machine_column, duration_column]
       :param job_column: Name of the output job ID column
       :param routing_column: Name of the routing template ID column
       :param operation_column: Name of the operation index column
       :param machine_column: Name of the machine column
       :param duration_column: Name of the processing time column
       :param offset: Starting number for new job IDs
       :param shuffle: Whether to shuffle routing templates before assignment
       :param seed: Random seed for shuffling
       :return: DataFrame with generated jssp [job_column, routing_column, operation_column, machine_column, duration_column]
       """

    # 1) Group routing template
    groups = [grp for _, grp in df_routings.groupby(routing_column, sort=False)]

    # 2) Optionally shuffle the routing templates
    if shuffle:
        random.seed(seed)
        random.shuffle(groups)

    # 3) Generate jssp from the routing templates
    new_recs = []
    for i, grp in enumerate(groups):
        job_id = offset + i
        routing_id = grp[routing_column].iloc[0]
        for _, row in grp.iterrows():
            new_recs.append({
                job_column: f"J25-{job_id:04d}",
                routing_column: routing_id,
                operation_column: row[operation_column],
                machine_column: row[machine_column],
                duration_column: row[duration_column]
            })

    return pd.DataFrame(new_recs).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------------

def get_df_with_ready_time_by_shift(
        df_arrivals: pd.DataFrame, shift_length: int = 480, arrival_column: str = "Arrival",
        ready_time_column: str = "Ready Time") -> pd.DataFrame:
    """
    Returns a copy of the DataFrame with an additional column containing the next shift-aligned ready time.

    :param df_arrivals: Input DataFrame containing arrival times.
    :param shift_length: Length of one shift in minutes. Default is 480 (8 hours).
    :param arrival_column: Name of the column that contains arrival times.
    :param ready_time_column: Name of the new column to store the calculated ready times.
    :return: A new DataFrame with the added ready time column.
    """
    df = df_arrivals.copy()
    df[ready_time_column] = np.ceil((df[arrival_column] + 1) / shift_length) * shift_length
    df[ready_time_column] = df[ready_time_column].astype(int)
    return df