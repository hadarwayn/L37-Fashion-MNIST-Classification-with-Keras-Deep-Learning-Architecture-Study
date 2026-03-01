"""
Training plots — accuracy and loss curves for each model.

These plots show how the model learns over time (epochs).
A healthy curve rises for accuracy and drops for loss.
If training and validation curves diverge, that's overfitting!
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file saving
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from src.utils.paths import get_graphs_dir
from src.data_loader import CLASS_NAMES

# Plot settings
DPI = 300
FIGSIZE_CURVES = (12, 5)
FIGSIZE_GRID = (15, 8)


def plot_training_curves(
    history: Dict[str, list],
    model_name: str,
    output_path: Path | None = None,
) -> None:
    """Plot accuracy and loss vs. epoch (side by side).

    Args:
        history: Keras history.history dict.
        model_name: Used in title and filename.
        output_path: Where to save the PNG. Auto-generated if None.
    """
    if output_path is None:
        safe = model_name.lower().replace(" ", "_").replace("+", "")
        output_path = get_graphs_dir() / f"{safe}_curves.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_CURVES)

    epochs = range(1, len(history["accuracy"]) + 1)
    best_epoch = int(np.argmin(history["val_loss"])) + 1

    # Accuracy
    ax1.plot(epochs, history["accuracy"], "b-", label="Training")
    ax1.plot(epochs, history["val_accuracy"], "r-", label="Validation")
    ax1.axvline(best_epoch, color="green", linestyle="--", alpha=0.5,
                label=f"Best epoch ({best_epoch})")
    ax1.set_title(f"{model_name} — Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(epochs, history["loss"], "b-", label="Training")
    ax2.plot(epochs, history["val_loss"], "r-", label="Validation")
    ax2.axvline(best_epoch, color="green", linestyle="--", alpha=0.5,
                label=f"Best epoch ({best_epoch})")
    ax2.set_title(f"{model_name} — Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_sample_images(x_data, y_data, n_per_class: int = 3) -> None:
    """Plot a grid of sample images — n_per_class images from each class.

    Saved to results/graphs/sample_images_grid.png.
    """
    fig, axes = plt.subplots(len(CLASS_NAMES), n_per_class,
                             figsize=(n_per_class * 2, len(CLASS_NAMES) * 2))
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        indices = np.where(y_data == cls_idx)[0][:n_per_class]
        for j, idx in enumerate(indices):
            ax = axes[cls_idx, j]
            img = x_data[idx]
            if img.ndim == 3:
                img = img.squeeze(-1)
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if j == 0:
                ax.set_title(cls_name, fontsize=9, loc="left")

    plt.suptitle("Fashion-MNIST — Sample Images (3 per class)", fontsize=14)
    plt.tight_layout()
    out = get_graphs_dir() / "sample_images_grid.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_class_distribution(y_data, split_name: str = "Training") -> None:
    """Bar chart of class counts. Saved to results/graphs/class_distribution.png."""
    counts = [int(np.sum(y_data == i)) for i in range(len(CLASS_NAMES))]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(CLASS_NAMES, counts, color=plt.cm.tab10.colors)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha="center", va="bottom", fontsize=9)

    ax.set_title(f"Fashion-MNIST {split_name} Set — Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out = get_graphs_dir() / "class_distribution.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")
