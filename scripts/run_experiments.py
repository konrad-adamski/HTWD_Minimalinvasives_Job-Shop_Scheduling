from __future__ import annotations

import argparse
from decimal import Decimal
from itertools import product
import tomllib  # Python 3.11+

from config.project_config import get_config_path
from src.Logger import Logger
from src.domain.Initializer import ExperimentInitializer
from src.domain.Query import ExperimentQuery
from src.runner import run_shifts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and run experiments from experiments_config.toml in CONFIG_PATH."
    )
    parser.add_argument(
        "--util",
        required=True,
        help=(
            'Must be "all" or a single numeric value from '
            '"max_bottleneck_utilization_list" in the config file.'
        ),
    )
    args = parser.parse_args()

    # Load config
    config_path = get_config_path("experiments_config.toml", as_string=False)
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    grid = cfg["grid"]
    run_cfg = cfg["run"]

    source_name: str = run_cfg["source_name"]
    shift_length: int = int(run_cfg["shift_length"])
    total_shift_number: int = int(run_cfg["total_shift_number"])

    all_utils: list[float] = grid["max_bottleneck_utilization_list"]

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
        if util_value not in all_utils:
            raise SystemExit(
                f"Error: --util must be 'all' or one of these values: {all_utils}"
            )
        selected_utils = [util_value]

    # Generate combinations and run
    for (util, a_lat, i_tar, sigma) in product(
        selected_utils,
        grid["absolute_lateness_ratio"],
        grid["inner_tardiness_ratio"],
        grid["simulation_sigma"],
    ):
        experiment_id = ExperimentInitializer.insert_experiment(
            source_name=source_name,
            absolute_lateness_ratio=a_lat,
            inner_tardiness_ratio=i_tar,
            max_bottleneck_utilization=Decimal(f"{util:.2f}"),
            sim_sigma=sigma,
        )
        logger_name = f"experiments_{util:.2f}"
        logger = Logger(name=logger_name, log_file=f"{logger_name}.log")
        run_shifts(
            experiment_id = experiment_id,
            shift_length = shift_length,
            total_shift_number = total_shift_number,
            logger = logger,
        )


if __name__ == "__main__":
    main()

