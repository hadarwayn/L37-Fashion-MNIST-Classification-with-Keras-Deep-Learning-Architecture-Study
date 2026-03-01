"""
Confusion matrix heatmaps — shows which classes get mixed up.

The diagonal (top-left to bottom-right) shows correct predictions.
Off-diagonal cells are mistakes. Bright off-diagonal cells reveal
which classes the model confuses — like T-shirt vs Shirt!
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from src.utils.paths import get_graphs_dir
from src.data_loader import CLASS_NAMES

DPI = 300
FIGSIZE = (10, 8)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    accuracy: float | None = None,
    output_path: Path | None = None,
) -> None:
    """Plot a confusion matrix heatmap with counts and percentages.

    Args:
        y_true: True integer labels.
        y_pred: Predicted integer labels.
        model_name: For title and filename.
        accuracy: Overall accuracy (added to title if given).
        output_path: Save location. Auto-generated if None.
    """
    from sklearn.metrics import confusion_matrix as cm_fn

    cm = cm_fn(y_true, y_pred)
    cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    # Annotations: count + percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0.5)

    title = f"Confusion Matrix — {model_name}"
    if accuracy is not None:
        title += f" (Accuracy: {accuracy:.2%})"
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path is None:
        safe = model_name.lower().replace(" ", "_").replace("+", "")
        output_path = get_graphs_dir() / f"{safe}_confusion.png"

    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_misclassified(
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    n: int = 20,
    output_path: Path | None = None,
) -> None:
    """Grid of images the model got wrong, showing true vs predicted labels.

    Args:
        x_test: Test images.
        y_true: True labels (integers).
        y_pred: Predicted labels (integers).
        model_name: For title and filename.
        n: Number of misclassified images to show.
        output_path: Save location.
    """
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        print(f"  {model_name}: No misclassifications found!")
        return
    sample = wrong_idx[:n]

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
    axes = axes.flatten()

    for i, idx in enumerate(sample):
        img = x_test[idx]
        if img.ndim == 3:
            img = img.squeeze(-1)
        axes[i].imshow(img, cmap="gray")
        true_name = CLASS_NAMES[y_true[idx]]
        pred_name = CLASS_NAMES[y_pred[idx]]
        axes[i].set_title(f"True: {true_name}\nPred: {pred_name}",
                          fontsize=7, color="red")
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(len(sample), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Misclassified Examples — {model_name}", fontsize=13)
    plt.tight_layout()

    if output_path is None:
        safe = model_name.lower().replace(" ", "_").replace("+", "")
        output_path = get_graphs_dir() / f"misclassified_{safe}.png"

    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")
