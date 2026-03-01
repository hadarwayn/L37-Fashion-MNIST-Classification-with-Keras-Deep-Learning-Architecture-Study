"""
Group A: Fully Connected (FC) Models — Models 1, 2, 3.

These models treat the image as a flat list of 784 numbers.
They have NO idea that pixels next to each other are related.
Think of it like reading a book by throwing all the letters in a bag
and trying to guess the story — you lose all the structure!

We test three FC designs to see how depth vs. width matters.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Named constants — no magic numbers
FC_INPUT_SHAPE = (28, 28)
NUM_CLASSES = 10

# Model 1: single hidden layer
BASELINE_UNITS = 128

# Model 2: narrow tower
NARROW_UNITS = 64
NARROW_DEPTH = 3  # hidden layers

# Model 3: wide single layer
WIDE_UNITS = 512


def build_fc_baseline(loss: str = "sparse_categorical_crossentropy") -> Sequential:
    """Model 1 — FC Baseline: the simplest possible neural network.

    Architecture: Flatten → Dense(128, ReLU) → Dense(10, Softmax)

    Like reading a scrambled book — you see every letter but lose
    the page layout. Our 'control group' that everything else
    must beat. Expected accuracy: ~85-87%.
    """
    model = Sequential([
        Flatten(input_shape=FC_INPUT_SHAPE),
        Dense(BASELINE_UNITS, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax"),
    ], name="model_01_fc_baseline")

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def build_narrow_deep_fc(loss: str = "sparse_categorical_crossentropy") -> Sequential:
    """Model 2 — Narrow Deep FC: a thin, tall stack of layers.

    Architecture: Flatten → Dense(64) × 3 → Dense(10)

    Like a long, thin pipe — information squeezes through many stages
    but each stage can only handle a little at a time.
    Tests whether depth helps FC networks. Expected accuracy: ~84-87%.
    """
    model = Sequential(name="model_02_narrow_deep_fc")
    model.add(Flatten(input_shape=FC_INPUT_SHAPE))
    for _ in range(NARROW_DEPTH):
        model.add(Dense(NARROW_UNITS, activation="relu"))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def build_wide_shallow_fc(loss: str = "sparse_categorical_crossentropy") -> Sequential:
    """Model 3 — Wide Shallow FC: one huge layer with many neurons.

    Architecture: Flatten → Dense(512, ReLU) → Dense(10, Softmax)

    Like a wide highway with lots of lanes but only one exit —
    lots of capacity but only one processing step.
    Tests whether brute-force width beats depth. Expected accuracy: ~86-88%.
    """
    model = Sequential([
        Flatten(input_shape=FC_INPUT_SHAPE),
        Dense(WIDE_UNITS, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax"),
    ], name="model_03_wide_shallow_fc")

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model
