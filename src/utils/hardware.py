"""
Hardware detection — reports what CPU/GPU is available.

Knowing your hardware matters because training on a GPU can be
5–50× faster than on a CPU. This module auto-detects and prints
a clear report so you know exactly what machine you're using.
"""

import platform
import sys
from typing import Dict


def get_hardware_info() -> Dict[str, str]:
    """Detect and return hardware details as a dictionary.

    Returns:
        Dict with keys: tf_version, gpu_available, gpu_name,
        python_version, os, cpu.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    gpu_available = len(gpus) > 0
    gpu_name = gpus[0].name if gpu_available else "None (CPU only)"

    # Try to get more detailed GPU info
    device_name = ""
    try:
        device_name = tf.test.gpu_device_name() or "CPU"
    except Exception:
        device_name = "CPU"

    return {
        "tf_version": tf.__version__,
        "gpu_available": str(gpu_available),
        "gpu_name": gpu_name,
        "gpu_device": device_name,
        "python_version": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
    }


def print_hardware_report() -> None:
    """Print a formatted hardware report to the console."""
    info = get_hardware_info()

    print("\n" + "=" * 55)
    print("  HARDWARE REPORT")
    print("=" * 55)
    print(f"  Python version   : {info['python_version']}")
    print(f"  TensorFlow       : {info['tf_version']}")
    print(f"  OS               : {info['os']}")
    print(f"  CPU              : {info['cpu']}")
    print(f"  GPU available    : {info['gpu_available']}")
    print(f"  GPU name         : {info['gpu_name']}")
    print(f"  GPU device       : {info['gpu_device']}")
    print("=" * 55 + "\n")


def get_device_strategy() -> str:
    """Return a short string describing the compute device.

    Used in results tables and log messages.
    """
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return f"GPU ({gpus[0].name})"
    return "CPU"
