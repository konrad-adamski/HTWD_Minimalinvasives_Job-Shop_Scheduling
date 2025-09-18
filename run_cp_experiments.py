from __future__ import annotations

import argparse
from decimal import Decimal
from itertools import product

try:
    import tomllib as pytoml
except ModuleNotFoundError:
    import tomli as pytoml

from config.project_config import get_config_path
from src.Logger import Logger
from src.domain.Initializer import ExperimentInitializer
from src.CP_Experiment_Runner import run_experiment


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
        "--time_limit",
        type=int,
        required=True,
        help="Solver time limit in seconds.",
    )
    parser.add_argument(
        "--bound_no_improvement_time",
        type=int,
        required=True,
        help="Time in seconds with no bound improvement before stopping.",
    )
    parser.add_argument(
        "--bound_warmup_time",
        type=int,
        required=True,
        help="Warmup time in seconds before bound improvement checks start.",
    )
    # NEU: Sigma kommt von der CLI (nicht mehr aus der Grid-Kombination)
    parser.add_argument(
        "--sigma",
        type=float,
        required=True,
        help='Simulation noise sigma. Must be one of "simulation_sigma" from the config file.',
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
    all_sigmas: list[float] = grid["simulation_sigma"]  # nur zur Validierung

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

    # Validate --sigma (nur validieren, NICHT kombinieren)
    if not _is_in_list_tol(args.sigma, all_sigmas):
        raise SystemExit(
            f"Error: --sigma must be one of these values from config: {all_sigmas}"
        )
    sigma = float(args.sigma)

    # Generate combinations (ohne sigma) und run
    for (util, a_lat, i_tar) in product(
        selected_utils,
        grid["absolute_lateness_ratio"],
        grid["inner_tardiness_ratio"],
    ):
        experiment_id = ExperimentInitializer.insert_experiment(
            source_name=source_name,
            absolute_lateness_ratio=a_lat,
            inner_tardiness_ratio=i_tar,
            max_bottleneck_utilization=Decimal(f"{util:.2f}"),
            sim_sigma=sigma,  # feste Sigma aus CLI
            experiment_type="CP",
        )
        logger_name = f"experiments_{util:.2f}_sig{sigma:g}"
        logger = Logger(name=logger_name, log_file=f"{logger_name}.log")
        run_experiment(
            experiment_id=experiment_id,
            shift_length=shift_length,
            total_shift_number=total_shift_number,
            logger=logger,
            time_limit=args.time_limit,
            bound_no_improvement_time=args.bound_no_improvement_time,
            bound_warmup_time=args.bound_warmup_time,
        )


if __name__ == "__main__":
    """
    Example usage:
    python run_cp_experiments.py --util 0.75 --sigma 0.1 --time_limit 1800 --bound_no_improvement_time 600 --bound_warmup_time 60
    python run_cp_experiments.py --util all --time_limit 900 --bound_no_improvement_time 300 --bound_warmup_time 30 --sigma 0.05
    """
    main()
