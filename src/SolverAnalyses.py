import re
from pathlib import Path
from typing import Union

import pandas as pd


class ConvergenceAnalysis:
    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated.")

    @staticmethod
    def _parse_cp_sat_bound_log(
            file_path: Union[str, Path], time_key: str = "Time",
            bestsol_key: str = "BestSol") -> list[dict]:
        """
        Parse OR-Tools CP-SAT log file for bound updates (#Bound lines).

        :param file_path: Path to the CP-SAT solver log file.
        :param time_key: Key name for elapsed time values in the returned dictionaries.
        :param bestsol_key: Key name for best solution values in the returned dictionaries.
        :return: List of dictionaries with extracted time and best solution values, excluding 'inf' entries.
        :rtype: list[dict]
        """
        file_path = Path(file_path)

        bound_lines = []
        with open(file_path, "r") as file:
            for line in file:
                if line.lstrip().startswith("#Bound"):
                    bound_lines.append(line.rstrip())

        if not bound_lines:
            return []

        pattern = re.compile(r"#Bound\s+([\d.]+)s\s+best:([\dinf]+)")

        records: list[dict] = []
        for line in bound_lines:
            m = pattern.search(line)
            if not m:
                continue
            time_sec = float(m.group(1))
            raw_best = m.group(2).lower()
            best_val = None if raw_best in {"inf", "-inf"} else int(raw_best)
            records.append({time_key: time_sec, bestsol_key: best_val})

        return records

    @classmethod
    def parse_cp_sat_bound_log_to_dataframe(
            cls, file_path: Union[str, Path], time_col: str = "Time", bestsol_col: str = "BestSol") -> pd.DataFrame:
        """
        Parse OR-Tools CP-SAT log file for bound updates (#Bound lines).

        :param file_path: Path to the CP-SAT solver log file.
        :param time_col: Column name for elapsed time values in the returned DataFrame.
        :param bestsol_col: Column name for best solution values in the returned DataFrame.
        :return: DataFrame containing extracted time and best solution values, filtered for non-NaN best solutions.
        :rtype: pd.DataFrame
        """
        records = cls._parse_cp_sat_bound_log(file_path=file_path, time_key=time_col, bestsol_key=bestsol_col)
        if not records:
            return pd.DataFrame(columns=[time_col, bestsol_col])

        df = pd.DataFrame(records)
        df = df[df[bestsol_col].notna()].reset_index(drop=True)
        return df
