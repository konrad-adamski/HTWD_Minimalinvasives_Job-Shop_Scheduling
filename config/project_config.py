from pathlib import Path
from typing import Optional, Union

# Basis variables (environment)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data"
SOLVER_LOGS_PATH = PROJECT_ROOT / "data/solver_logs"

CONFIG_PATH = PROJECT_ROOT / "config"


def get_data_path(
        sub_directory: Optional[str] = None, file_name: Optional[str] = None,
        as_string: bool = False) -> Union[Path, str]:
    """
    Return an absolute path inside the /data directory.

    :param sub_directory: Optional subdirectory name inside /data. Must exist if given.
    :param file_name: Optional file name to append to the path. No existence check performed.
    :param as_string: If True, return the result as a string instead of a Path object.
    :return: Absolute path as a Path or string.
    :raises FileNotFoundError: If sub_directory is provided but does not exist.
    """
    if sub_directory:
        dir_path = DATA_PATH / sub_directory.lstrip("/\\")
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
    else:
        dir_path = DATA_PATH

    target = dir_path / file_name if file_name else dir_path

    return str(target) if as_string else target


def get_config_path(file_name: Optional[str] = None,as_string: bool = False):
    dir_path = CONFIG_PATH
    target = dir_path / file_name if file_name else dir_path
    return str(target) if as_string else target

def get_solver_logs_path(
        sub_directory: Optional[str] = None,
        file_name: Optional[str] = None,
        as_string: bool = False
) -> Union[Path, str]:
    """
    Return an absolute path inside the /solver_logs directory.

    :param sub_directory: Optional subdirectory name inside /solver_logs.
                          If it does not exist, it will be created.
    :param file_name: Optional file name to append to the path.
                      No existence check performed.
    :param as_string: If True, return the result as a string instead of a Path object.
    :return: Absolute path as a Path or string.
    """
    if sub_directory:
        dir_path = SOLVER_LOGS_PATH / sub_directory.lstrip("/\\")
        dir_path.mkdir(parents=True, exist_ok=True)  # erstellen falls fehlt
    else:
        dir_path = SOLVER_LOGS_PATH

    target = dir_path / file_name if file_name else dir_path
    return str(target) if as_string else target


# Beispiel
if __name__ == "__main__":
    print(get_data_path())                 # → PROJECT_ROOT/data
   # print(get_data_path("logs"))           # → PROJECT_ROOT/data/logs
   # print(get_data_path("logs/run1.txt"))  # → PROJECT_ROOT/data/logs/run1.txt
