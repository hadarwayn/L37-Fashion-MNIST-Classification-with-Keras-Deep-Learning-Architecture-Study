"""
Comparison charts — grand accuracy, time, and FC-vs-CNN visualisations.

The 'hero chart' is the grand accuracy bar chart: all 10 models
sorted best-to-worst, colour-coded by group.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

from src.utils.paths import get_graphs_dir

DPI = 300
GROUP_COLORS = {"A": "#4C72B0", "B": "#55A868", "C": "#DD8452", "D": "#C44E52"}


def plot_grand_accuracy_comparison(results: List[Dict[str, Any]]) -> None:
    """THE hero chart — all models' test accuracy, sorted best to worst."""
    sorted_r = sorted(results, key=lambda r: r["test_accuracy"], reverse=True)
    names = [f"M{r['model_id']}: {r['model_name']}" for r in sorted_r]
    accs = [r["test_accuracy"] for r in sorted_r]
    colors = [GROUP_COLORS.get(r["group"], "gray") for r in sorted_r]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(names, accs, color=colors)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{acc:.2%}", va="center", fontsize=9)

    ax.set_xlabel("Test Accuracy")
    ax.set_title("Grand Comparison — All 10 Architectures (sorted)")
    ax.set_xlim(0.75, 1.0)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend = [Patch(color=c, label=f"Group {g}") for g, c in GROUP_COLORS.items()]
    ax.legend(handles=legend, loc="lower right")

    plt.tight_layout()
    out = get_graphs_dir() / "grand_comparison_accuracy.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_grand_time_comparison(results: List[Dict[str, Any]]) -> None:
    """Bar chart of training time per model."""
    names = [f"M{r['model_id']}: {r['model_name']}" for r in results]
    times = [r["training_time_seconds"] for r in results]
    colors = [GROUP_COLORS.get(r["group"], "gray") for r in results]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(names, times, color=colors)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{t:.0f}s", va="center", fontsize=9)

    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Training Time Comparison — All 10 Models")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = get_graphs_dir() / "grand_comparison_time.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_fc_vs_cnn(results: List[Dict[str, Any]]) -> None:
    """Grouped bar chart: FC models (1-3) vs CNN models (4-10)."""
    fc = [r for r in results if r["type"] == "FC"]
    cnn = [r for r in results if r["type"] == "CNN"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x_fc = np.arange(len(fc))
    x_cnn = np.arange(len(cnn)) + len(fc) + 1

    ax.bar(x_fc, [r["test_accuracy"] for r in fc],
           color=GROUP_COLORS["A"], label="FC Models")
    ax.bar(x_cnn, [r["test_accuracy"] for r in cnn],
           color=GROUP_COLORS["B"], label="CNN Models")
    ax.set_xticks(list(x_fc) + list(x_cnn))
    ax.set_xticklabels([f"M{r['model_id']}" for r in fc] +
                       [f"M{r['model_id']}" for r in cnn])
    ax.set_ylabel("Test Accuracy")
    ax.set_title("FC vs CNN — The Definitive Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.75, 1.0)
    plt.tight_layout()
    out = get_graphs_dir() / "fc_vs_cnn_comparison.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# Re-export from split file to keep one public API
from src.visualization.comparison_extra import (  # noqa: E402
    plot_loss_function_comparison, plot_group_summary,
)

__all__ = [
    "plot_grand_accuracy_comparison", "plot_grand_time_comparison",
    "plot_fc_vs_cnn", "plot_loss_function_comparison", "plot_group_summary",
]
