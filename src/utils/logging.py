from typing import Dict, Any


def print_experiment_log(experiment_log: Dict[str, Dict[str, Any]]) -> None:
    """
    Pretty-prints a nested experiment log dictionary in a structured format.

    :param experiment_log: Dictionary with nested structure like:
                           {
                               "experiment_info": {...},
                               "experiment_config": {...},
                               "model_info": {...},
                               "solver_info": {...}
                           }
    """
    print("\n===== EXPERIMENT LOG SUMMARY =====")
    for section_name, section_dict in experiment_log.items():
        print(f"[{section_name.upper()}]")
        for key, value in section_dict.items():
            print(f"  {key:50}: {value}")
    print("==================================\n")
