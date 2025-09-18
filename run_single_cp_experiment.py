from __future__ import annotations

import argparse
from decimal import Decimal

from src.CP_Experiment_Runner import run_experiment
from src.Logger import Logger
from src.domain.Initializer import ExperimentInitializer


def config_experiment(
    source_name: str,
    absolute_lateness_ratio: float,
    inner_tardiness_ratio: float,
    max_bottleneck_utilization: float,
    sim_sigma: float,
) -> int:
    """
    Legt ein Experiment mit den übergebenen Parametern in der DB an
    und gibt die experiment_id zurück.
    """
    experiment_id = ExperimentInitializer.insert_experiment(
        source_name=source_name,
        absolute_lateness_ratio=absolute_lateness_ratio,
        inner_tardiness_ratio=inner_tardiness_ratio,
        max_bottleneck_utilization=Decimal(f"{max_bottleneck_utilization:.2f}"),
        sim_sigma=sim_sigma,
    )
    return experiment_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single experiment with optional parameters.")
    parser.add_argument("--util", type=float, default=1.00, help="Max bottleneck utilization (default: 1.00)")
    parser.add_argument("--lateness_ratio", type=float, default=0.5, help="Absolute lateness ratio (default: 0.5)")
    parser.add_argument("--tardiness_ratio", type=float, default=0.5, help="Inner tardiness ratio (default: 0.5)")
    parser.add_argument("--sim_sigma", type=float, default=0.25, help="Simulation sigma (default: 0.25)")

    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"⚠️ Ignoring unknown arguments: {unknown}")

    util = args.util
    lateness_ratio = args.lateness_ratio
    tardiness_ratio = args.tardiness_ratio
    sim_sigma = args.sim_sigma

    # 1) Experiment anlegen
    experiment_id = config_experiment(
        source_name="Fisher and Thompson 10x10",
        absolute_lateness_ratio=lateness_ratio,
        inner_tardiness_ratio=tardiness_ratio,
        max_bottleneck_utilization=util,
        sim_sigma=sim_sigma,
    )

    # 2) Logger
    logger_name = f"single_experiment_{util:.2f}"
    logger = Logger(name=logger_name, log_file=f"{logger_name}.log")

    # 3) Experiment starten
    run_experiment(
        experiment_id=experiment_id,
        shift_length=1440,
        total_shift_number=20,
        logger=logger,
        time_limit=60 * 60* 24,
        bound_warmup_time= 60 * 60 * 6,
        bound_no_improvement_time=60 * 60* 6
    )

"""terminal
python run_single_cp_experiment.py
"""

"""terminal
python run_single_cp_experiment.py --util 1.0 --lateness_ratio 0.5 --tardiness_ratio 0.5 --sim_sigma 0.2
"""
