from typing import Optional, Literal

import pandas as pd
import numpy as np


def generate_arrivals_from_mean_interarrival_time(
        job_number: int, mean_interarrival_time: float,
        start_time: float = 0.0, var_type: Literal['Integer', 'Float'] = 'Integer',
        random_seed: Optional[int] = 120) -> np.ndarray:
    """
    Generate a list of job arrival times based on a given mean interarrival time.

    Arrival times are drawn from an exponential distribution and accumulated over time.

    :param job_number: Number of arrivals to generate
    :param mean_interarrival_time: Expected interarrival time (mean of exponential distribution)
    :param start_time: Time of the first possible arrival
    param var_type: 'Integer' for minute-precision timestamps, 'Float' for 2-decimal float values
    :param random_seed: Optional seed for reproducibility (set to None for nondeterministic behavior)
    :return: Numpy array of arrival times
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # 1) Generate exponentially distributed interarrival times
    interarrival_times = np.random.exponential(scale=mean_interarrival_time, size=job_number)
    interarrival_times[0] = 0  # first arrival exactly at start_time

    # 2) Calculate cumulative arrival times
    arrivals = start_time + np.cumsum(interarrival_times)

    # 3) Round depending on output type
    if var_type == 'Integer':
        arrivals = np.floor(arrivals).astype(int)
    elif var_type == 'Float':
        arrivals = np.round(arrivals, 2)
    else:
        raise ValueError(f"Invalid var_type: {var_type}. Must be 'Integer' or 'Float'.")

    return arrivals


def calculate_mean_interarrival_time(
        df_routings: pd.DataFrame, u_b_mmax: float = 0.9, routing_column: str = 'Routing_ID',
        machine_column: str = "Machine", duration_column: str = "Processing Time", verbose: bool = False) -> float:
    """
    Calculate the mean interarrival time t_a that results in a target utilization
    (u_b_mmax < 1.0) of the bottleneck machine.

    :param df_routings: DataFrame containing routing definitions with machine and processing times
    :param u_b_mmax: Target utilization of the bottleneck machine (e.g., 0.9)
    :param routing_column: Column identifying distinct routings
    :param machine_column: Column identifying machines
    :param duration_column: Column with processing times
    :param verbose: If True, prints bottleneck machine info and processing time vector
    :return: Mean interarrival time t_a, rounded to 2 decimal places
    """

    # 1) Uniform routing probability
    n_routings = df_routings[routing_column].nunique()
    p = [1.0 / n_routings] * n_routings

    # 2) Get vector of processing times on the bottleneck machine
    vec_t_b_mmax = _get_vec_t_b_mmax(
        df_routings=df_routings,
        routing_column=routing_column,
        machine_column=machine_column,
        duration_column=duration_column,
        verbose=verbose
    )

    if verbose:
        print(f"Processing times on bottleneck machine: {vec_t_b_mmax}")

    # 3) Calculate mean interarrival time using expected processing time and utilization
    t_a = sum(p[i] * vec_t_b_mmax[i] for i in range(n_routings)) / u_b_mmax
    return round(t_a, 2)


def _get_vec_t_b_mmax(
        df_routings: pd.DataFrame, routing_column: str = 'Routing_ID',
        machine_column: str = "Machine",duration_column: str = "Processing Time", verbose=False):
    """
    Return the processing time of each routing on the bottleneck machine.

    The list is ordered according to the first appearance of routing IDs in the DataFrame.

    :param df_routings: DataFrame containing routing, machine, and duration information
    :param routing_column: Column identifying each routing (template or job structure)
    :param machine_column: Column identifying the machines
    :param duration_column: Column containing processing times
    :param verbose: If True, prints machine usage information
    :return: List of processing times on the bottleneck machine (0 if not used)
    """
    # Determine the bottleneck machine
    eng = _get_engpassmaschine(
        df_routings=df_routings,
        machine_column=machine_column,
        duration_column=duration_column,
        verbose=verbose
    )

    # Preserve original routing order
    routing_order = df_routings[routing_column].unique().tolist()

    # Get processing time on the bottleneck machine for each routing
    proc_on_eng = df_routings[df_routings[machine_column] == eng].set_index(routing_column)[duration_column].to_dict()

    # Return a list of durations in the original routing order, or 0 if not used
    return [proc_on_eng.get(routing, 0) for routing in routing_order]


def _get_engpassmaschine(
        df_routings: pd.DataFrame,
        machine_column: str = "Machine",
        duration_column: str = "Processing Time",
        verbose=False):
    """
    Identify the bottleneck machine with the highest total processing time.

    :param df_routings: DataFrame containing routing structure with machine and duration info
    :param machine_column: Column name identifying the machines
    :param duration_column: Column name for processing time
    :param verbose: If True, prints total load per machine
    :return: Label of the bottleneck machine (same type as machine_column)
    """
    # Compute total processing time per machine
    usage = df_routings.groupby(machine_column)[duration_column].sum().to_dict()

    # Optional output for inspection
    if verbose:
        print("Machine load (total processing time):")
        for m, total in sorted(usage.items(), key=lambda x: str(x[0])):
            print(f"  {m}: {total}")

    # Return the machine with the highest total usage
    return max(usage, key=usage.get)