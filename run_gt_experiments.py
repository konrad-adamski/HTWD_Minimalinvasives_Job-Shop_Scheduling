from __future__ import annotations

import argparse
from decimal import Decimal
from itertools import product

from src.GT_Experiment import init_experiment

try:
    import tomllib as pytoml
except ModuleNotFoundError:
    import tomli as pytoml

from config.project_config import get_config_path


def _is_in_list_tol(x: float, values: list[float], tol: float = 1e-12) -> bool:
    """Mit Toleranz prüfen, ob x in values (Float-Robustheit)."""
    return any(abs(x - v) <= tol for v in values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and run experiments from experiments_config.toml in CONFIG_PATH."
    )
    parser.add_argument(
        "--util",
        required=True,
        help=(
            'Must be "all" or a single numeric value from '
            '"max_bottleneck_utilization" in the config file.'
        ),
    )
    parser.add_argument(
        "--sigma",
        required=True,
        help=(
            'Must be "all" or a single numeric value from '
            '"simulation_sigma" in the config file.'
        ),
    )
    parser.add_argument(
        "--priority_rule",
        required=True,
        help="Priority rule to use (string, no validation).",
    )

    args = parser.parse_args()

    # Load config
    config_path = get_config_path("experiments_config.toml", as_string=False)
    with open(config_path, "rb") as f:
        cfg = pytoml.load(f)

    grid = cfg["grid"]
    run_cfg = cfg["run"]

    source_name: str = run_cfg["source_name"]
    shift_length: int = int(run_cfg["shift_length"])
    total_shift_number: int = int(run_cfg["total_shift_number"])

    all_utils: list[float] = grid["max_bottleneck_utilization"]
    all_sigmas: list[float] = grid["simulation_sigma"]

    # Validate --util
    if args.util.lower() == "all":
        selected_utils = list(all_utils)
    else:
        try:
            util_value = float(args.util)
        except ValueError:
            raise SystemExit(
                f'Error: --util must be "all" or a single numeric value from {all_utils}.'
            )
        if not _is_in_list_tol(util_value, all_utils):
            raise SystemExit(
                f"Error: --util must be 'all' or one of these values: {all_utils}"
            )
        selected_utils = [util_value]

    # Validate --sigma
    if args.sigma.lower() == "all":
        selected_sigmas = list(all_sigmas)
    else:
        try:
            sigma_value = float(args.sigma)
        except ValueError:
            raise SystemExit(
                f'Error: --sigma must be "all" or a single numeric value from {all_sigmas}.'
            )
        if not _is_in_list_tol(sigma_value, all_sigmas):
            raise SystemExit(
                f"Error: --sigma must be 'all' or one of these values: {all_sigmas}"
            )
        selected_sigmas = [sigma_value]

    # Generate combinations (jetzt util × sigma)
    for util, sigma in product(selected_utils, selected_sigmas):
        init_experiment(
            shift_length=shift_length,
            total_shift_number=total_shift_number,
            priority_rule=args.priority_rule,   # <-- direkt aus CLI
            source_name=source_name,
            max_bottleneck_utilization=Decimal(f"{util:.2f}"),
            sim_sigma=float(sigma),
        )


if __name__ == "__main__":
    """
    Example usage:
    python run_gt_experiments.py --util 0.75 --sigma 0.1 --priority_rule SLACK
    python run_gt_experiments.py --util all --sigma 0.2 --priority_rule DEVIATION
    python run_gt_experiments.py --util all --sigma all --priority_rule DEVIATION_INSERT
    """
    main()
