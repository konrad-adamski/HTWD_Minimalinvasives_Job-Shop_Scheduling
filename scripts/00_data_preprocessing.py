import json

from src.DataPreprocessor import DataPreprocessor
from configs.path_manager import get_path

if __name__ == "__main__":
    basic_data_path = get_path("data", "basic")
    input_file_path = basic_data_path / "jobshop.txt"
    output_file_path = basic_data_path / "jobshop_instances.json"

    # Transform text file
    instances_dict = DataPreprocessor.transform_file_to_instances_dictionary(input_file_path)

    print("Instances", "-"*90)
    for key in instances_dict.keys():
        print(key)

    # Save JSON
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(instances_dict, f, indent=2)   # type: ignore

    # Example
    print("\nFisher Thompson 10x10 [machine, duration]", "-" * 58)
    ft_10_instance = instances_dict["instance ft10"]
    for key, value in ft_10_instance.items():
        print(f"{key}: {value}")
    print("-" * 100, end="\n\n")