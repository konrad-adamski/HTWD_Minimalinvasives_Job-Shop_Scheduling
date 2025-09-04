import math
import random
import re
from typing import Literal, Optional

from src.domain.orm_models import JobOperation


def get_time_str(minutes_in: float) -> str:
    """
    Converts a time value in minutes to an absolute timestamp string (Day HH:MM:SS).

    :param minutes_in: Time value in minutes (can be fractional).
    :return: Time string in absolute format.
    """
    return format_time(minutes_in, time_type="absolute_time")

def get_duration(minutes_in: float) -> str:
    """
    Converts a time value in minutes to a human-readable duration string.

    :param minutes_in: Time value in minutes (can be fractional).
    :return: Time string as duration (e.g., '05 minutes 30 seconds').
    """
    return format_time(minutes_in, time_type="time_difference")

def format_time(minutes_in: float, time_type: Literal["absolute_time", "time_difference"] = "absolute_time") -> str:
    """
    Formats a time value in minutes either as an absolute timestamp or as a duration.

    :param minutes_in: Time value in minutes (can be fractional).
    :param time_type: Type of formatting. Use "absolute_time" for 'Day HH:MM:SS',
                      or "time_difference" for 'MM minutes SS seconds'.
    :return: Formatted time string.
    :raises ValueError: If an unknown time_type is provided.
    """
    minutes_total = int(minutes_in)
    seconds = int(round((minutes_in - minutes_total) * 60))

    if time_type == "absolute_time":
        days = minutes_total // 1440  # 1440 Minuten pro Tag
        remaining_minutes = minutes_total % 1440
        hours = remaining_minutes // 60
        minutes = remaining_minutes % 60
        return f"Day {days} {hours:02}:{minutes:02}:{seconds:02}"

    elif time_type == "time_difference":
        parts = []
        if minutes_total:
            parts.append(f"{minutes_total:02} minute{'s' if minutes_total != 1 else ''}")
        if seconds:
            parts.append(f"{seconds:02} second{'s' if seconds != 1 else ''}")
        return " ".join(parts) if parts else "0 seconds"

    else:
        raise ValueError(f"Unknown time_type: {time_type}")


def duration_log_normal_by_vc(duration: float, variation: float = 0.2, round_decimal: int = 0, seed: Optional[int] = None) -> float:
    """
    Returns a lognormally distributed duration based on a given coefficient of variation (CV).
    The distribution is constructed so that its expected value in real space is approximately equal to `duration`.

    :param duration: Expected mean duration in real space.
    :param variation: Coefficient of variation (CV), e.g. 0.2 for 20% relative spread.
    :param round_decimal: Number of decimal places to round the result.
    :param seed: Optional random seed for reproducibility.
    :return: A sampled duration from the log-normal distribution.
    """
    variation = max(variation, 0.0)
    sigma = math.sqrt(math.log(1 + variation ** 2))
    mu = math.log(duration) - 0.5 * sigma ** 2

    rng = random.Random(seed) if seed is not None else random
    result = rng.lognormvariate(mu, sigma)

    return round(result, round_decimal)


def duration_log_normal(duration: float, sigma: float = 0.2, round_decimal: int = 0, seed: Optional[int] = None) -> float:
    """
    Returns a log-normal distributed duration whose expected value in real space
    is approximately equal to the given `duration`.

    :param duration: Expected mean duration in real space.
    :param sigma: Standard deviation in log space (must be non-negative).
    :param round_decimal: Number of decimal places to round the result.
    :param seed: Optional random seed for reproducibility.
    :return: A sampled duration from the log-normal distribution.
    """
    sigma =max(sigma, - sigma)

    rnd_numb_gen = random.Random(seed) if seed is not None else random

    # Compute mu so that the expected value in real space is approximately equal to `duration` (E[X] ≈ duration)
    mu = math.log(duration) - 0.5 * sigma ** 2

    result = rnd_numb_gen.lognormvariate(mu, sigma)

    return round(result, round_decimal)


def _seed_from_op(op: JobOperation) -> int:
    """
    Builds a deterministic seed from (Job, Operation).

    Expected job format: <value1>-<value2>-<value3> (e.g. 01-07500-05).
    Uses only <value1><value3> (e.g. 0105).

    :param op: Operation instance.
    :return: Deterministic integer seed.
    """
    job_id = str(op.job_id)

    # Extract first and last numeric group
    parts = re.findall(r"\d+", job_id)
    if len(parts) >= 2:
        job_digits = f"{parts[0]}{parts[-1]}"
    else:
        job_digits = "".join(parts)  # Fallback: take everything

    # Ensure position_number is always 2 digits (01, 02, ..., 10, 11, ...)
    pos_str = str(op.position_number).zfill(2)

    return int(f"{job_digits}{pos_str}")


def simulated_duration(op: JobOperation, sigma: float = 0.2, round_decimal: int = 0, extra_seed: int = 0) -> float:
    """
    Generates a simulated duration for an operation via ``duration_log_normal``.

    Uses a stable seed based on (Job, Operation).

    :param op: Operation instance.
    :param sigma: Standard deviation in log space.
    :param round_decimal: Number of decimal places to round the result.
    :param extra_seed: Optional additional offset added to the base seed.
    """
    duration = op.duration
    seed = _seed_from_op(op)
    return duration_log_normal(duration, sigma=sigma, round_decimal=round_decimal, seed=seed + extra_seed)

if __name__ == "__main__":
    print(duration_log_normal(100, 0.1))
    print(duration_log_normal_by_vc(100, 0.5))