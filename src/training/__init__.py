"""Training pipeline — model training, evaluation, loss functions, results I/O."""

from src.training.trainer import train_model, train_all_models
from src.training.loss_functions import (
    get_sparse_categorical_crossentropy,
    get_categorical_crossentropy,
    L2RegularizedLoss,
    build_l2_model_variant,
)
from src.training.results_io import save_results_csv, print_summary_table

__all__ = [
    "train_model", "train_all_models",
    "get_sparse_categorical_crossentropy", "get_categorical_crossentropy",
    "L2RegularizedLoss", "build_l2_model_variant",
    "save_results_csv", "print_summary_table",
]
