"""
Results I/O — save training results to CSV and print summary tables.

Keeps main.py focused on orchestration by handling all
results persistence and console reporting here.
"""

import csv
from typing import List, Dict, Any

from src.utils.paths import get_tables_dir


def save_results_csv(results: List[Dict[str, Any]]) -> None:
    """Save a summary CSV with one row per model."""
    out = get_tables_dir() / "results_summary.csv"
    fields = [
        "model_id", "model_name", "group", "type", "test_accuracy",
        "test_loss", "training_time_seconds", "epochs_trained",
        "best_epoch", "total_params",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\n  Results CSV saved: {out}")


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """Print a final summary table to the console."""
    print(f"\n{'='*80}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    hdr = (f"{'ID':>3} {'Model':<25} {'Group':>5} "
           f"{'Acc':>8} {'Loss':>8} {'Time':>8} {'Params':>10}")
    print(hdr)
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["test_accuracy"], reverse=True):
        print(
            f"{r['model_id']:>3} {r['model_name']:<25} {r['group']:>5} "
            f"{r['test_accuracy']:>8.4f} {r['test_loss']:>8.4f} "
            f"{r['training_time_seconds']:>7.1f}s {r['total_params']:>10,}"
        )
    print(f"{'='*80}\n")
