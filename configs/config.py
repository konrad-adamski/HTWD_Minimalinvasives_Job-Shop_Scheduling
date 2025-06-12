import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

with open(PROJECT_ROOT / "configs" / "paths.yaml") as f:
    _paths = yaml.safe_load(f)


def get_path(*keys: str) -> Path | None:
    """
    Returns an absolute path from paths.yaml using nested keys.
    If the result is a dictionary, it prints available subkeys and returns None.
    """
    node = _paths
    for key in keys:
        if key not in node:
            print(f"Key '{key}' not found.")
            return None
        node = node[key]

    if isinstance(node, dict):
        print(f"'{' → '.join(keys)}' is a group. Available subkeys: {', '.join(node.keys())}")
        return None

    return PROJECT_ROOT / Path(node)


def list_keys(*keys: str) -> list[str] | None:
    """
    Returns a list of available subkeys at the specified key path.
    Example: list_keys("data") → ["basic", "processed"]
    """
    node = _paths
    for key in keys:
        if key not in node:
            print(f"Key '{key}' not found.")
            return None
        node = node[key]

    if isinstance(node, dict):
        return list(node.keys())
    else:
        print(f"'{' → '.join(keys)}' is not a group with subkeys.")
        return None

