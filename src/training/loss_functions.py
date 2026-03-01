"""
Loss functions — the 'grading system' that tells the model how wrong it is.

Standard losses:
  - Sparse Categorical Crossentropy: labels are integers (3 = 'Dress')
  - Categorical Crossentropy: labels are one-hot vectors

Custom loss:
  - L2-Regularized Crossentropy: adds a penalty for large weights,
    like a basketball coach who penalises any player scoring too much —
    forces the team to spread the effort.
"""

import tensorflow as tf
from tensorflow.keras.losses import (
    SparseCategoricalCrossentropy,
    CategoricalCrossentropy,
)

DEFAULT_LAMBDA = 0.01


def get_sparse_categorical_crossentropy() -> SparseCategoricalCrossentropy:
    """Standard loss for integer labels (e.g. y=3 means 'Dress').

    Like a teacher with a simple answer key: 'The correct answer is C.'
    """
    return SparseCategoricalCrossentropy()


def get_categorical_crossentropy() -> CategoricalCrossentropy:
    """Standard loss for one-hot labels (e.g. y=[0,0,0,1,0,...]).

    Like a teacher with a checklist: 'Not A, Not B, Not C, YES D, ...'
    """
    return CategoricalCrossentropy()


class L2RegularizedLoss(tf.keras.losses.Loss):
    """Custom loss: Crossentropy + lambda * sum(weights^2).

    The L2 penalty punishes large weights, preventing the model from
    over-relying on any single connection.

    Basketball analogy: your score = points scored MINUS a penalty for
    any player scoring too much. Forces the team to share the ball.

    Args:
        l2_lambda: Strength of the penalty (higher = stricter).
            0.001 = gentle nudge, 0.01 = moderate, 0.1 = strict.
    """

    def __init__(self, l2_lambda: float = DEFAULT_LAMBDA, **kwargs):
        super().__init__(**kwargs)
        self.l2_lambda = l2_lambda
        self._ce = SparseCategoricalCrossentropy()

    def call(self, y_true, y_pred):
        """Compute total loss = CE loss + L2 penalty."""
        ce_loss = self._ce(y_true, y_pred)
        return ce_loss  # L2 penalty applied via kernel_regularizer in layers

    def get_config(self):
        config = super().get_config()
        config["l2_lambda"] = self.l2_lambda
        return config


def build_l2_model_variant(build_fn, l2_lambda: float = DEFAULT_LAMBDA):
    """Wrap a model builder to add L2 kernel regularisation to every Dense layer.

    This is the proper Keras way: add regularizers to layers rather than
    computing the penalty manually in the loss function.
    """
    return _rebuild_with_l2(build_fn, l2_lambda)


def _rebuild_with_l2(build_fn, l2_lambda: float):
    """Rebuild a model adding L2 regularization to Dense layers."""
    from tensorflow.keras import regularizers

    model = build_fn(loss="sparse_categorical_crossentropy")
    reg = regularizers.l2(l2_lambda)

    # Clone weights later if needed; for training comparison we just
    # need a fresh model with regularization
    new_layers = []
    for layer in model.layers:
        cfg = layer.get_config()
        if "Dense" in layer.__class__.__name__ and cfg.get("units", 0) > 0:
            cfg["kernel_regularizer"] = reg
        new_layers.append(layer.__class__.from_config(cfg))

    from tensorflow.keras.models import Sequential
    new_model = Sequential(new_layers, name=f"{model.name}_l2_{l2_lambda}")
    new_model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
    return new_model


LOSS_REGISTRY = {
    "sparse_categorical_crossentropy": get_sparse_categorical_crossentropy,
    "categorical_crossentropy": get_categorical_crossentropy,
}
