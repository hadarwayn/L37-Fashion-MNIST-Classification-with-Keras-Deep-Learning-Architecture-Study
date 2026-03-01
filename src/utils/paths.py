"""
Path utilities — every path in the project resolved from one place.

Uses pathlib.Path for cross-platform compatibility (Windows + Linux).
No absolute paths anywhere — everything is relative to the project root.
"""

from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Works by going up from this file:
      src/utils/paths.py  →  src/utils/  →  src/  →  project root
    """
    return Path(__file__).resolve().parent.parent.parent


def get_src_dir() -> Path:
    """Return path to the src/ directory."""
    return get_project_root() / "src"


def get_config_path() -> Path:
    """Return path to config/training_config.yaml."""
    return get_project_root() / "config" / "training_config.yaml"


def get_results_dir() -> Path:
    """Return path to results/ and ensure it exists."""
    path = get_project_root() / "results"
    path.mkdir(exist_ok=True)
    return path


def get_graphs_dir() -> Path:
    """Return path to results/graphs/ and ensure it exists."""
    path = get_results_dir() / "graphs"
    path.mkdir(exist_ok=True)
    return path


def get_tables_dir() -> Path:
    """Return path to results/tables/ and ensure it exists."""
    path = get_results_dir() / "tables"
    path.mkdir(exist_ok=True)
    return path


def get_examples_dir() -> Path:
    """Return path to results/examples/ and ensure it exists."""
    path = get_results_dir() / "examples"
    path.mkdir(exist_ok=True)
    return path


def get_logs_dir() -> Path:
    """Return path to logs/ and ensure it exists."""
    path = get_project_root() / "logs"
    path.mkdir(exist_ok=True)
    return path


def get_log_config_path() -> Path:
    """Return path to logs/config/log_config.json."""
    return get_project_root() / "logs" / "config" / "log_config.json"
