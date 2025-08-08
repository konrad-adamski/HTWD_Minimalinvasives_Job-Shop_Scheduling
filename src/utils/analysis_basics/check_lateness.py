import numpy as np
import pandas as pd
from typing import Literal

def get_jobs_aggregated(
        df: pd.DataFrame,
        column: Literal['Lateness', 'Tardiness', 'Earliness'] = 'Lateness',
        steps: int = 60, min_val: int = -120, max_val: int = 120,
        right_closed: bool = False) -> pd.DataFrame:
    """
    Aggregates a column of job metrics (e.g., Tardiness or Lateness) into labeled bins.
    Returns a one-row DataFrame with bin labels as columns and counts as values.

    :param df: Input DataFrame containing the column to aggregate.
    :param column: Metric to aggregate. Must be one of: 'Lateness', 'Tardiness', 'Earliness'.
    :param steps: Width of each bin.
    :param min_val: Minimum value for binning.
    :param max_val: Maximum absolute value for binning.
    :param right_closed: Whether bins include the right edge (default: False).
    :return: A one-row DataFrame with bin labels as column names and counts as values.
    """

    # 1. Check that the specified column exists
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist. Available columns: {list(df.columns)}")

    # 2. Define bin boundaries
    if column in ['Tardiness', 'Earliness']:
        min_val = 0
        inner_bins = list(range(min_val, max_val + steps, steps))
        bins = inner_bins + [np.inf]  # no -inf for always-positive metrics
    else:
        inner_bins = list(range(min_val, max_val + steps, steps))
        if 0 not in inner_bins:
            inner_bins.append(0)
            inner_bins = sorted(inner_bins)
        bins = [-np.inf] + inner_bins + [np.inf]

    # 3. Count zero values separately
    zero_count = (df[column] == 0).sum()
    non_zero = df.loc[df[column] != 0, column]

    # 4. Define bin labels and sorting keys
    labels = []
    bin_keys = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if np.isneginf(lo):
            labels.append(f"<{int(hi)}")
            bin_keys.append(lo + 0.1)
        elif np.isposinf(hi):
            labels.append(f">{int(lo)}")
            bin_keys.append(hi - 0.1)
        else:
            labels.append(f"{int(lo)} - {int(hi)}")
            bin_keys.append((lo + hi) / 2)

    # 5. Apply binning to non-zero values
    grouped = pd.cut(non_zero, bins=bins, labels=labels, right=right_closed, include_lowest=True)
    counts = grouped.value_counts().reindex(labels, fill_value=0)

    # 6. Add the count for exact zero
    counts["0"] = zero_count
    bin_keys.append(0)
    labels.append("0")

    # 7. Sort bins using sort keys
    sort_df = pd.DataFrame({f'{column}_Interval': labels, 'key': bin_keys}).set_index(f'{column}_Interval')
    sorted_counts = counts.loc[sort_df.sort_values('key').index]

    # 8. Return result as a single-row DataFrame
    return pd.DataFrame([sorted_counts])



