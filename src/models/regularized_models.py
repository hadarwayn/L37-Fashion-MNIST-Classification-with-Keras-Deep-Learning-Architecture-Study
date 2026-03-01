"""
Group C: Regularized CNN Models — Models 8, 9.

Both models use the SAME architecture as Model 4 (Baseline CNN)
with one added technique. This isolates the effect of the
regularization method — a controlled scientific experiment.

Model 8 (Dropout): Randomly 'turns off' neurons during training,
  like a coach benching random star players so the whole team
  gets stronger.

Model 9 (BatchNorm): Normalizes data between layers, like a
  quality check between each step in a factory assembly line.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization,
)

CNN_INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)

# Same architecture as Model 4
BASE_FILTERS = [32, 64]
DENSE_UNITS = 128

# Dropout rates
DROPOUT_CONV = 0.25   # After each conv block
DROPOUT_DENSE = 0.5   # Before final output layer


def build_cnn_dropout(loss: str = "sparse_categorical_crossentropy") -> Sequential:
    """Model 8 — CNN + Dropout: prevents memorisation by randomly disabling neurons.

    Architecture: Same as Model 4 but with Dropout(0.25) after each
    conv block and Dropout(0.5) before the final Dense layer.

    Like a soccer team where the coach randomly benches 25% of players
    during practice — forces every player to step up.
    Expected accuracy: ~89-92% (better generalisation than Model 4).
    """
    model = Sequential([
        Conv2D(BASE_FILTERS[0], KERNEL_SIZE, activation="relu",
               input_shape=CNN_INPUT_SHAPE),
        MaxPooling2D(POOL_SIZE),
        Dropout(DROPOUT_CONV),

        Conv2D(BASE_FILTERS[1], KERNEL_SIZE, activation="relu"),
        MaxPooling2D(POOL_SIZE),
        Dropout(DROPOUT_CONV),

        Flatten(),
        Dense(DENSE_UNITS, activation="relu"),
        Dropout(DROPOUT_DENSE),
        Dense(NUM_CLASSES, activation="softmax"),
    ], name="model_08_cnn_dropout")

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def build_cnn_batchnorm(loss: str = "sparse_categorical_crossentropy") -> Sequential:
    """Model 9 — CNN + BatchNorm: normalises data between every layer.

    Architecture: Same as Model 4 but with BatchNormalization after
    every Conv2D and Dense layer.

    Like a factory quality check between each production step — make
    sure the half-finished product is in good shape before moving on.
    Typically reaches peak accuracy 30-50% faster than Model 4.
    Expected accuracy: ~90-92%.
    """
    model = Sequential([
        Conv2D(BASE_FILTERS[0], KERNEL_SIZE, activation="relu",
               input_shape=CNN_INPUT_SHAPE),
        BatchNormalization(),
        MaxPooling2D(POOL_SIZE),

        Conv2D(BASE_FILTERS[1], KERNEL_SIZE, activation="relu"),
        BatchNormalization(),
        MaxPooling2D(POOL_SIZE),

        Flatten(),
        Dense(DENSE_UNITS, activation="relu"),
        BatchNormalization(),
        Dense(NUM_CLASSES, activation="softmax"),
    ], name="model_09_cnn_batchnorm")

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model
