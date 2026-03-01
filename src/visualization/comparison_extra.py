"""
Extra comparison charts — loss function and per-group summaries.

Split from comparison_charts.py to stay under the 150-line limit.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from src.utils.paths import get_graphs_dir

DPI = 300
GROUP_COLORS = {"A": "#4C72B0", "B": "#55A868", "C": "#DD8452", "D": "#C44E52"}


def plot_loss_function_comparison(loss_results: List[Dict[str, Any]]) -> None:
    """Overlay accuracy curves for same model with different loss functions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for r in loss_results:
        label = r.get("loss_name", "unknown")
        h = r["history"]
        epochs = range(1, len(h["accuracy"]) + 1)
        ax1.plot(epochs, h["val_accuracy"], label=label)
        ax2.plot(epochs, h["val_loss"], label=label)

    ax1.set_title("Loss Function Comparison — Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Loss Function Comparison — Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = get_graphs_dir() / "loss_function_comparison.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_group_summary(
    results: List[Dict[str, Any]], group: str,
) -> None:
    """Comparison chart for models within one group."""
    group_r = [r for r in results if r["group"] == group]
    if not group_r:
        return
    names = [f"M{r['model_id']}: {r['model_name']}" for r in group_r]
    accs = [r["test_accuracy"] for r in group_r]

    fig, ax = plt.subplots(figsize=(10, 5))
    color = GROUP_COLORS.get(group, "gray")
    bars = ax.bar(names, accs, color=color)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{acc:.2%}", ha="center", fontsize=10)

    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Group {group} Summary")
    ax.set_ylim(0.75, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    out = get_graphs_dir() / f"group_{group.lower()}_summary.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")
