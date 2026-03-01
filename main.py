"""
L37 — Fashion-MNIST Deep Learning Architecture Study.

Entry point for local execution. Trains up to 10 neural network
architectures on Fashion-MNIST and generates all results.

Usage:
    python main.py                  # Train all 10 models
    python main.py --models 1,4,8   # Train selected models only
    python main.py --epochs 10      # Override default epochs
    python main.py --skip-viz       # Skip visualisation generation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_fashion_mnist, print_class_distribution
from src.models import get_all_model_ids
from src.models.cnn_models import build_baseline_cnn
from src.training.trainer import train_model, train_all_models
from src.training.loss_functions import build_l2_model_variant
from src.training.results_io import save_results_csv, print_summary_table
from src.visualization.training_plots import (
    plot_training_curves, plot_sample_images, plot_class_distribution,
)
from src.visualization.confusion_matrix import plot_confusion_matrix, plot_misclassified
from src.visualization.comparison_charts import (
    plot_grand_accuracy_comparison, plot_grand_time_comparison,
    plot_fc_vs_cnn, plot_loss_function_comparison, plot_group_summary,
)
from src.utils.logger import setup_logger, print_log_status
from src.utils.hardware import print_hardware_report

logger = setup_logger("main")


def parse_args():
    p = argparse.ArgumentParser(description="Fashion-MNIST Architecture Study")
    p.add_argument("--models", type=str, default="all",
                   help="Comma-separated model IDs (1-10) or 'all'")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--skip-viz", action="store_true")
    return p.parse_args()


def run_loss_comparison(data: dict, epochs: int, batch_size: int) -> list:
    """Train Model 4 with different loss functions."""
    results = []
    for loss, name, onehot in [
        ("sparse_categorical_crossentropy", "Sparse CE", False),
        ("categorical_crossentropy", "Categorical CE", True),
    ]:
        m = build_baseline_cnn(loss=loss)
        r = train_model(m, data, 4, epochs, batch_size, use_onehot=onehot)
        r["loss_name"] = name
        results.append(r)

    m = build_l2_model_variant(build_baseline_cnn, l2_lambda=0.01)
    r = train_model(m, data, 4, epochs, batch_size, use_onehot=False)
    r["loss_name"] = "L2 (λ=0.01)"
    results.append(r)
    return results


def generate_visualisations(results, loss_results, data):
    """Generate all plots from training results."""
    for r in results:
        plot_training_curves(r["history"], r["model_name"])
        plot_confusion_matrix(r["y_true"], r["y_pred"],
                              r["model_name"], r["test_accuracy"])

    best = max(results, key=lambda r: r["test_accuracy"])
    worst = min(results, key=lambda r: r["test_accuracy"])
    plot_misclassified(data["x_test_cnn"], best["y_true"],
                       best["y_pred"], best["model_name"])
    plot_misclassified(data["x_test_cnn"], worst["y_true"],
                       worst["y_pred"], worst["model_name"])

    if len(results) > 1:
        plot_grand_accuracy_comparison(results)
        plot_grand_time_comparison(results)
        plot_fc_vs_cnn(results)
        for g in ["A", "B", "C", "D"]:
            plot_group_summary(results, g)
    if loss_results:
        plot_loss_function_comparison(loss_results)


def main() -> None:
    args = parse_args()
    print_hardware_report()

    print("\n[1/6] Loading Fashion-MNIST dataset...")
    data = load_fashion_mnist()
    print_class_distribution(data)

    if not args.skip_viz:
        print("\n[2/6] Generating data visualisations...")
        plot_sample_images(data["x_train"], data["y_train"])
        plot_class_distribution(data["y_train"])

    model_ids = get_all_model_ids() if args.models == "all" else [
        int(x.strip()) for x in args.models.split(",")]
    epochs = args.epochs or 20

    print(f"\n[3/6] Training {len(model_ids)} model(s)...")
    results = train_all_models(model_ids, data, epochs, args.batch_size)

    loss_results = []
    if 4 in model_ids and not args.skip_viz:
        print("\n[4/6] Running loss function comparison...")
        loss_results = run_loss_comparison(data, epochs, args.batch_size)

    if not args.skip_viz and results:
        print("\n[5/6] Generating visualisations...")
        generate_visualisations(results, loss_results, data)

    print("\n[6/6] Saving results...")
    if results:
        save_results_csv(results)
        print_summary_table(results)

    print_log_status()
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
