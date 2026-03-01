"""
Ring buffer logging system — keeps log files from growing forever.

Think of it like a circular notebook: when you reach the last page,
you start overwriting from the first page. This way logs never eat
up all your disk space.

Implements the course-mandated ring buffer with configurable
max_lines and max_files.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.paths import get_log_config_path, get_logs_dir


class RingBufferHandler(logging.FileHandler):
    """File handler that rotates when a file exceeds max_lines."""

    def __init__(
        self,
        log_dir: Path,
        prefix: str = "fashion_mnist",
        ext: str = ".log",
        max_lines: int = 1000,
        max_files: int = 5,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._prefix = prefix
        self._ext = ext
        self._max_lines = max_lines
        self._max_files = max_files
        self._current_lines = 0
        self._file_index = 0

        self._log_dir.mkdir(parents=True, exist_ok=True)
        filepath = self._make_path()
        super().__init__(str(filepath), mode="a", encoding="utf-8")

    # --- internals ---------------------------------------------------

    def _make_path(self) -> Path:
        return self._log_dir / f"{self._prefix}_{self._file_index}{self._ext}"

    def _rotate(self) -> None:
        """Move to next file, wrapping around after max_files."""
        self.close()
        self._file_index = (self._file_index + 1) % self._max_files
        self._current_lines = 0
        new_path = self._make_path()
        # Overwrite old file when we wrap around
        self.baseFilename = str(new_path)
        self.stream = self._open()

    def emit(self, record: logging.LogRecord) -> None:
        """Write one log record, rotating if the file is full."""
        if self._current_lines >= self._max_lines:
            self._rotate()
        super().emit(record)
        self._current_lines += 1


def _load_log_config() -> dict:
    """Load log configuration from JSON file."""
    config_path = get_log_config_path()
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Sensible fallback if config file is missing
    return {
        "ring_buffer": {"max_lines_per_file": 1000, "max_files": 5,
                        "file_prefix": "fashion_mnist", "file_extension": ".log"},
        "format": {"pattern": "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
                   "date_format": "%Y-%m-%d %H:%M:%S"},
        "levels": {"console": "INFO", "file": "DEBUG"},
    }


def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """Create and return a configured logger with console + ring-buffer file output."""
    cfg = _load_log_config()
    rb = cfg["ring_buffer"]
    fmt_cfg = cfg["format"]
    levels = cfg["levels"]

    logger = logging.getLogger(name or "fashion_mnist")
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt_cfg["pattern"], datefmt=fmt_cfg["date_format"])

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, levels["console"].upper()))
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Ring buffer file handler
    fh = RingBufferHandler(
        log_dir=get_logs_dir(),
        prefix=rb.get("file_prefix", "fashion_mnist"),
        ext=rb.get("file_extension", ".log"),
        max_lines=rb.get("max_lines_per_file", 1000),
        max_files=rb.get("max_files", 5),
    )
    fh.setLevel(getattr(logging, levels["file"].upper()))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def print_log_status() -> None:
    """Print a summary of existing log files."""
    logs_dir = get_logs_dir()
    log_files = sorted(logs_dir.glob("*.log"))
    print(f"\n{'='*50}")
    print(f"  Log Status -- {len(log_files)} file(s) in {logs_dir}")
    for lf in log_files:
        lines = sum(1 for _ in open(lf, encoding="utf-8"))
        size_kb = lf.stat().st_size / 1024
        print(f"  {lf.name:30s}  {lines:>5} lines  {size_kb:>6.1f} KB")
    print(f"{'='*50}\n")
