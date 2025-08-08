import re
import pandas as pd

from pathlib import Path
from typing import Dict, List


class DataPreprocessor:
    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated.")

    @staticmethod
    def _step1_exclude_initial_text(content: str, skip_until_marker: int = 1) -> str:
        """
        Removes the text up to and including the N-th line that contains multiple '+' characters.

        :param content: The full text content.
        :param skip_until_marker: Index of the '+++' line after which the text should be kept.
        :return: The remaining text starting after the specified marker line.
        """
        # Find all lines containing +++
        matches = list(re.finditer(r"\n.*\+{3,}.*\n", content))

        # Keep everything after the N-th +++ line
        return content[matches[skip_until_marker].end():]

    @staticmethod
    def _step2_parse_text_with_instances_to_dict(content: str, verbose: bool = False) -> dict:
        """
        Parses a structured text with alternating instance names and data blocks into a dictionary.

        :param content: A string containing instance descriptions and matrix blocks separated by '+++' lines.
        :param verbose: If True, enables debug output (optional).
        :return: A dictionary where keys are instance descriptions
            and values are the corresponding matrix blocks (as strings).
        """

        # Separate blocks using +++ lines and remove unnecessary spaces
        raw_blocks = [block.strip() for block in re.split(r"\n.*\+{3,}.*\n", content) if block.strip()]

        if verbose:
            print("====== Raw blocks example ======")
            for i, b in enumerate(raw_blocks[:4]):
                print(f"--- {b} ---\n") if i % 2 == 0 else print(b, "\n")
            print("=" * 20)

        # Ensure that the number of blocks is even
        if len(raw_blocks) % 2 != 0:
            raise ValueError("Number of blocks is odd – each instance requires exactly 2 blocks (description + matrix)")

        # Build dictionary
        instance_dict = {}

        for i in range(0, len(raw_blocks), 2):
            key = raw_blocks[i].strip()  # e.g. "instance abz5"
            lines = raw_blocks[i + 1].splitlines()  # contains matrix block including matrix-info
            cleaned_lines = lines[2:]  # remove matrix-info (e.g. 10 10)
            matrix_block = "\n".join(cleaned_lines)  # reassemble the matrix
            instance_dict[key] = matrix_block

        return instance_dict


    @staticmethod
    def _step3_structure_dict(raw_dict: Dict[str, str]) -> Dict[str, Dict[int, List[Dict[str, int]]]]:
        """
        :param raw_dict: Dictionary mapping instance names to whitespace-separated job routing strings.
        :return: Nested dictionary with structured operation dictionaries.
        """
        structured_dict = {}
        for instance_name, matrix_text in raw_dict.items():
            lines = matrix_text.strip().splitlines()
            jobs = {}
            for job_id, line in enumerate(lines):
                try:
                    numbers = list(map(int, line.strip().split()))
                    job_ops = [{"machine": numbers[i], "duration": numbers[i + 1]} for i in range(0, len(numbers), 2)]
                    jobs[job_id] = job_ops
                except ValueError:
                    continue
            structured_dict[instance_name] = jobs
        return structured_dict

    @classmethod
    def transform_file_to_instances_dictionary(cls, file_path: Path) -> dict:
        # Read file
        file = open(file_path, encoding="utf-8")
        content = file.read()
        file.close()

        content_without_introduction = cls._step1_exclude_initial_text(content)

        # Dictionary with instances as keys and matrix as value (string)
        instances_string_dict = cls._step2_parse_text_with_instances_to_dict(content_without_introduction)

        # Dictionary with instances as keys and matrix as value (dictionary/JSON of routings)
        return cls._step3_structure_dict(instances_string_dict)


    @staticmethod
    def routing_dict_to_df(
            routings_dict: dict, routing_column: str = 'Routing_ID', operation_column: str = 'Operation',
            machine_column: str = "Machine", duration_column: str = "Processing Time",) -> pd.DataFrame:
        """
        Converts a routing dictionary with structured operations into a pandas DataFrame.

        :param routings_dict: Dictionary where each key is a routing ID (e.g., 0, 1, 2)
                              and each value is a list of operations as {"machine": int, "duration": int}.
        :param routing_column: Name of the column that will store the routing ID.
        :param operation_column: Name of the column that will store the operation index.
        :param machine_column: Name of the column that will store the machine name (e.g., ``'M00'``).
        :param duration_column: Name of the column that will store the processing time.

        :return: DataFrame with one row per operation, including routing ID, operation index, machine, and processing time.
        """
        records = []
        for plan_id, ops in routings_dict.items():
            for op_idx, op in enumerate(ops):
                records.append({
                    routing_column: plan_id,
                    operation_column: op_idx,
                    machine_column: f'M{op["machine"]:02d}',
                    duration_column: op["duration"]
                })
        df = pd.DataFrame(records, columns=[routing_column, operation_column, machine_column, duration_column])
        return df


