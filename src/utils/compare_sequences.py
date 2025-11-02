import editdistance
import pandas as pd

from statistics import mean
from scipy.stats import kendalltau
from typing import Dict, List, Tuple, Optional


def compute_sum_levenshtein_distance(
        previous_schedule: pd.DataFrame, new_schedule: pd.DataFrame,
        comparison_start_time: float) -> int:
    """
    Computes the total Levenshtein distance across all machines between
    the job sequences of the original and revised schedules after a given start time.

    The sequences are symmetrically filtered to include only jobs that are common
    to both versions per machine.

    :param previous_schedule: The original schedule DataFrame with 'Job', 'Machine', and 'Start' columns.
    :param new_schedule: The revised schedule DataFrame in the same format.
    :param comparison_start_time: Start time threshold; only jobs with start >= this value are considered.

    :return: The sum of Levenshtein distances across all machines.
    """
    machines, original_sequences, revised_sequences = _get_machines_and_sequences_dicts(
        previous_schedule, new_schedule, comparison_start_time
    )

    total_distance = 0
    for machine in machines:
        orig = original_sequences[machine]
        revised = revised_sequences[machine]
        distance = _compute_levenshtein_distance(orig, revised)
        total_distance += distance

    return total_distance


def compute_mean_kendall_tau(
        previous_schedule: pd.DataFrame, new_schedule: pd.DataFrame,
        comparison_start_time: float) -> Optional[float]:
    """
    Computes the mean Kendall's Tau across all machines between the job sequences
    of the original and revised schedules after a given start time.

    The sequences are symmetrically filtered to include only jobs that are common
    to both versions per machine.

    :param previous_schedule: The original schedule DataFrame with 'Job', 'Machine', and 'Start' columns.
    :param new_schedule: The revised schedule DataFrame in the same format.
    :param comparison_start_time: Start time threshold; only jobs with start >= this value are considered.

    :return: The mean Kendall's Tau across all machines, or None if no valid sequences are found.
    """
    machines, original_sequences, revised_sequences = _get_machines_and_sequences_dicts(
        previous_schedule, new_schedule, comparison_start_time
    )

    taus = []
    for machine in machines:
        orig = original_sequences[machine]
        revised = revised_sequences[machine]
        tau = _compute_kendall_tau(orig, revised)
        if tau is not None:
            taus.append(tau)

    if not taus:
        return None  # No valid comparisons possible
    return mean(taus)


def has_sequence_changed(
        previous_schedule: pd.DataFrame, new_schedule: pd.DataFrame,
        comparison_start_time: float) -> bool:
    """
    Returns True if any machine has a changed job sequence after the given start time.

    :param previous_schedule: Original schedule with 'Job', 'Machine', 'Start'.
    :param new_schedule: Revised schedule in same format.
    :param comparison_start_time: Only jobs with start >= this time are considered.

    :return: True if at least one machine has a different job sequence, else False.
    """
    machines, original_sequences, revised_sequences = _get_machines_and_sequences_dicts(
        previous_schedule, new_schedule, comparison_start_time
    )

    for machine in machines:
        if original_sequences[machine] != revised_sequences[machine]:
            return True  # early exit on first change
    return False


def get_shared_operations_number(
        previous_schedule: pd.DataFrame, new_schedule: pd.DataFrame,
        comparison_start_time: float) -> int:
    """
    Computes the total number of shared operations across all machines after a given start time.
    Only operations that appear in both schedules per machine are counted.

    :param previous_schedule: The original schedule with 'Job', 'Machine', 'Start'.
    :param new_schedule: The revised schedule in same format.
    :param comparison_start_time: Threshold time; only operations with start >= this value are considered.

    :return: Total number of shared jobs (operations) across all common machines.
    """
    machines, original_sequences, _ = _get_machines_and_sequences_dicts(
        previous_schedule, new_schedule, comparison_start_time
    )

    # Since _get_machines_and_sequences_dicts returns symmetrically filtered sequences,
    # the length of original_sequences[machine] directly corresponds to the number of shared jobs
    return sum(len(original_sequences[machine]) for machine in machines)

def get_comparison_dataframe(
        previous_schedule: pd.DataFrame, new_schedule: pd.DataFrame,
        comparison_start_time: float) -> pd.DataFrame:
    """
    Builds a DataFrame comparing per-machine job sequences from two schedules after a given start time.

    The DataFrame includes the original and revised sequences, Levenshtein distance,
    and Kendall's Tau for each machine.

    :param previous_schedule: The original schedule with 'Job', 'Machine', 'Start'.
    :param new_schedule: The revised schedule in same format.
    :param comparison_start_time: Only jobs with start >= this value are considered.

    :return: DataFrame indexed by machine with sequence and comparison metrics.
    """
    machines, original_sequences, revised_sequences = _get_machines_and_sequences_dicts(
        previous_schedule, new_schedule, comparison_start_time
    )

    records = []
    for machine in machines:
        orig = original_sequences[machine]
        revised = revised_sequences[machine]

        record = {
            "Original Sequence": orig,
            "Revised Sequence": revised,
            "Levenshtein": _compute_levenshtein_distance(orig, revised),
            "Kendall Tau": _compute_kendall_tau(orig, revised)
        }
        records.append((machine, record))

    df = pd.DataFrame.from_dict(dict(records), orient='index')
    df.index.name = "Machine"
    return df

def _get_machines_and_sequences_dicts(
    previous_schedule: pd.DataFrame,
    new_schedule: pd.DataFrame,
    comparison_start_time: float
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Extracts per-machine job sequences from two schedules after a given start time.
    Only machines present in both schedules are included, and job sequences are
    filtered to include only jobs that appear in both versions per machine.

    :param previous_schedule: The original schedule DataFrame with 'Job', 'Machine', and 'Start' columns.
    :param new_schedule: The revised schedule DataFrame in the same format.
    :param comparison_start_time: Start time threshold; only jobs with start >= this value are considered.

    :returns: A tuple containing the list of common machines, the original job sequences per machine
        and the revised job sequences per machine. Sequences are symmetrically filtered to contain only shared jobs.
    """

    # Normalize column names
    df_prev = previous_schedule.rename(columns={'Start': 'Start_prev', 'Job': 'Job'})
    df_new  = new_schedule.rename(columns={'Start': 'Start_rev', 'Job': 'Job'})

    # Filter by start time
    df_prev = df_prev[df_prev['Start_prev'] >= comparison_start_time]
    df_new  = df_new[df_new['Start_rev'] >= comparison_start_time]

    # Sort by machine and start time
    df_prev_sorted = df_prev[['Machine', 'Start_prev', 'Job']].sort_values(['Machine', 'Start_prev'])
    df_new_sorted  = df_new[['Machine', 'Start_rev', 'Job']].sort_values(['Machine', 'Start_rev'])

    # Ensure job identifiers are strings (for safe comparison)
    df_prev_sorted['Job'] = df_prev_sorted['Job'].astype(str)
    df_new_sorted['Job'] = df_new_sorted['Job'].astype(str)

    # Group job sequences by machine
    prev_seqs = df_prev_sorted.groupby('Machine')['Job'].apply(list).to_dict()
    new_seqs  = df_new_sorted.groupby('Machine')['Job'].apply(list).to_dict()

    # Determine common machines
    machines = sorted(set(prev_seqs) & set(new_seqs))

    # Build symmetrically filtered sequences per machine
    original_sequences = {}
    revised_sequences = {}

    for machine in machines:
        orig_seq = prev_seqs[machine]
        new_seq = new_seqs[machine]

        shared_jobs = set(orig_seq) & set(new_seq)
        original_sequences[machine] = [job for job in orig_seq if job in shared_jobs]
        revised_sequences[machine]  = [job for job in new_seq if job in shared_jobs]

    return machines, original_sequences, revised_sequences


def _compute_kendall_tau(original_sequence: List[str], revised_sequence: List[str]) -> Optional[float]:
    """
    Computes Kendall's Tau between two job sequences with identical elements.

    :param original_sequence: Reference job sequence.
    :param revised_sequence: Job sequence to compare (same elements, different order).
    :return: Kendall's Tau correlation coefficient rounded to 4 decimals (or 1.0 if undefined).
    """
    if len(original_sequence) != len(revised_sequence) or len(original_sequence) < 2:
        return 1.0  # Invalid input or too short for Tau

    # Map each job to its rank in the original sequence
    rank_original = {job: i for i, job in enumerate(original_sequence)}

    # Translate revised_sequence into the corresponding ranks from original_sequence
    rank_revised = [rank_original[job] for job in revised_sequence]

    # Baseline represents the ideal order (0, 1, 2, ..., n-1)
    baseline = list(range(len(rank_revised)))

    # Compare the order in revised_sequence against the original using Kendall's Tau
    tau, _ = kendalltau(baseline, rank_revised)

    return round(tau, 4) if tau is not None else 1.0


def _compute_levenshtein_distance(original_sequence: List[str], revised_sequence: List[str]) -> int:
    """
    Computes the Levenshtein distance between two job sequences.

    :param original_sequence: Reference job sequence.
    :param revised_sequence: Job sequence to compare.
    :return: Minimum number of edit operations to transform original_sequence into revised_sequence.
    """
    return editdistance.eval(original_sequence, revised_sequence)