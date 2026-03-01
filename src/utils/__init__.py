"""Utility package — paths, logging, hardware detection."""

from src.utils.paths import (
    get_project_root, get_config_path, get_results_dir,
    get_graphs_dir, get_tables_dir, get_examples_dir, get_logs_dir,
)
from src.utils.logger import setup_logger, print_log_status
from src.utils.hardware import print_hardware_report, get_hardware_info

__all__ = [
    "get_project_root", "get_config_path", "get_results_dir",
    "get_graphs_dir", "get_tables_dir", "get_examples_dir", "get_logs_dir",
    "setup_logger", "print_log_status",
    "print_hardware_report", "get_hardware_info",
]
