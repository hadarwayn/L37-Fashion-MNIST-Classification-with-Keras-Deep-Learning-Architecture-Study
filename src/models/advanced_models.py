"""
Group D: Advanced Architecture — Model 10 (Skip Connections).

Skip connections let information 'jump over' layers using a shortcut.
Imagine a tall building with both stairs AND an elevator — even if
the stairs are slow or blocked, the elevator keeps things moving.

This is the core idea behind ResNet, which won the 2015 ImageNet
competition and changed deep learning forever. Without skip
connections, very deep networks suffer from 'vanishing gradients'
(the learning signal fades before reaching early layers).
"""

from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense,
    Add, BatchNormalization, Activation,
)
from tensorflow.keras.models import Model

CNN_INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
FILTERS = [32, 64, 128, 256]
DENSE_UNITS = 128


def _residual_block(x, filters: int):
    """One residual block: Conv → BN → ReLU → Conv → BN → Add → ReLU.

    If input and output dimensions differ, a 1×1 Conv2D 'projection
    shortcut' matches them so the Add layer works.
    """
    shortcut = x

    # Main path
    x = Conv2D(filters, KERNEL_SIZE, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, KERNEL_SIZE, padding="same")(x)
    x = BatchNormalization()(x)

    # Projection shortcut if channel counts differ
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Skip connection — the key idea!
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


def build_cnn_skip(loss: str = "sparse_categorical_crossentropy") -> Model:
    """Model 10 — CNN + Skip Connections (mini-ResNet).

    Architecture: 4 residual blocks (32→64→128→256 filters) with
    MaxPool between blocks, then Dense(128) → Dense(10).
    Uses Keras Functional API (not Sequential) because Add() needs
    two inputs.

    Like a building with stairs AND an elevator — information can
    take the shortcut if the regular path is too lossy. This solves
    the vanishing gradient problem that hurts Model 6.
    Expected accuracy: ~91-93%.
    """
    inputs = Input(shape=CNN_INPUT_SHAPE)
    x = inputs

    for f in FILTERS:
        x = _residual_block(x, f)
        x = MaxPooling2D(POOL_SIZE)(x)

    x = Flatten()(x)
    x = Dense(DENSE_UNITS, activation="relu")(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="model_10_cnn_skip")
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model
