"""
Model registry — a single place to look up any of the 10 architectures.

Usage:
    from src.models import get_model, MODEL_INFO, get_all_model_ids
    model = get_model(4, loss="sparse_categorical_crossentropy")
"""

from src.models.fc_models import (
    build_fc_baseline, build_narrow_deep_fc, build_wide_shallow_fc,
)
from src.models.cnn_models import (
    build_baseline_cnn, build_deep_cnn, build_very_deep_cnn, build_wide_cnn,
)
from src.models.regularized_models import build_cnn_dropout, build_cnn_batchnorm
from src.models.advanced_models import build_cnn_skip

# Maps model ID → builder function
MODEL_REGISTRY = {
    1: build_fc_baseline,
    2: build_narrow_deep_fc,
    3: build_wide_shallow_fc,
    4: build_baseline_cnn,
    5: build_deep_cnn,
    6: build_very_deep_cnn,
    7: build_wide_cnn,
    8: build_cnn_dropout,
    9: build_cnn_batchnorm,
    10: build_cnn_skip,
}

# Human-readable metadata for each model
MODEL_INFO = {
    1:  {"name": "FC Baseline",           "group": "A", "type": "FC",  "short": "fc_baseline"},
    2:  {"name": "Narrow Deep FC",        "group": "A", "type": "FC",  "short": "narrow_deep_fc"},
    3:  {"name": "Wide Shallow FC",       "group": "A", "type": "FC",  "short": "wide_shallow_fc"},
    4:  {"name": "Baseline CNN",          "group": "B", "type": "CNN", "short": "baseline_cnn"},
    5:  {"name": "Deep CNN",              "group": "B", "type": "CNN", "short": "deep_cnn"},
    6:  {"name": "Very Deep CNN",         "group": "B", "type": "CNN", "short": "very_deep_cnn"},
    7:  {"name": "Wide CNN",              "group": "B", "type": "CNN", "short": "wide_cnn"},
    8:  {"name": "CNN + Dropout",         "group": "C", "type": "CNN", "short": "cnn_dropout"},
    9:  {"name": "CNN + BatchNorm",       "group": "C", "type": "CNN", "short": "cnn_batchnorm"},
    10: {"name": "CNN + Skip Connections", "group": "D", "type": "CNN", "short": "cnn_skip"},
}


def get_model(model_id: int, loss: str = "sparse_categorical_crossentropy"):
    """Build and return a compiled Keras model by its ID (1-10)."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model ID {model_id}. Valid: 1-10.")
    return MODEL_REGISTRY[model_id](loss=loss)


def get_all_model_ids() -> list[int]:
    """Return sorted list of all available model IDs."""
    return sorted(MODEL_REGISTRY.keys())


__all__ = ["get_model", "get_all_model_ids", "MODEL_REGISTRY", "MODEL_INFO"]
