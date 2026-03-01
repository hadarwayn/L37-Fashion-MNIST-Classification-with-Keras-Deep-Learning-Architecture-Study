"""
Group B: CNN Models — Models 4, 5, 6, 7.

CNNs scan the image in small patches (like a magnifying glass)
so they understand that nearby pixels are related. This spatial
awareness is why CNNs crush FC models for image tasks.

We test four CNN designs to explore depth and width tradeoffs.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
)

CNN_INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)


def build_baseline_cnn(loss: str = "sparse_categorical_crossentropy") -> Sequential:
    """Model 4 — Baseline CNN: the standard recipe most tutorials use.

    Architecture: Conv(32) → Pool → Conv(64) → Pool → Dense(128) → Dense(10)

    Like reading a book properly: first letters, then words, then
    sentences. The 'aha!' moment — this should beat ALL FC models.
    Expected accuracy: ~89-91%.
    """
    model = Sequential([
        Conv2D(32, KERNEL_SIZE, activation="relu", input_shape=CNN_INPUT_SHAPE),
        MaxPooling2D(POOL_SIZE),
        Conv2D(64, KERNEL_SIZE, activation="relu"),
        MaxPooling2D(POOL_SIZE),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax"),
    ], name="model_04_baseline_cnn")

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def build_deep_cnn(loss: str = "sparse_categorical_crossentropy") -> Sequential:
    """Model 5 — Deep CNN: 4 conv blocks stacked for greater depth.

    Architecture: 4 × (Conv → Conv → Pool) with 32→64→128→256 filters,
    then Dense(128) → Dense(10).

    Like a detective examining evidence at higher and higher levels.
    Tests whether more layers = more accuracy. Expected accuracy: ~90-92%.
    """
    filters = [32, 64, 128, 256]
    model = Sequential(name="model_05_deep_cnn")
    for i, f in enumerate(filters):
        kw = {"input_shape": CNN_INPUT_SHAPE} if i == 0 else {}
        model.add(Conv2D(f, KERNEL_SIZE, activation="relu", padding="same", **kw))
        model.add(Conv2D(f, KERNEL_SIZE, activation="relu", padding="same"))
        model.add(MaxPooling2D(POOL_SIZE))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def build_very_deep_cnn(loss: str = "sparse_categorical_crossentropy") -> Sequential:
    """Model 6 — Very Deep CNN: 5 conv blocks pushing depth to the extreme.

    Architecture: 5 × (Conv → Conv → Pool) with 32→64→128→256→512 filters,
    same-padding to preserve dimensions.

    Like a bureaucracy with too many departments — at some point more
    levels slow things down. May show vanishing gradients!
    Expected accuracy: ~90-93% (or worse if gradients vanish).
    """
    filters = [32, 64, 128, 256, 512]
    model = Sequential(name="model_06_very_deep_cnn")
    for i, f in enumerate(filters):
        kw = {"input_shape": CNN_INPUT_SHAPE} if i == 0 else {}
        model.add(Conv2D(f, KERNEL_SIZE, activation="relu", padding="same", **kw))
        model.add(Conv2D(f, KERNEL_SIZE, activation="relu", padding="same"))
        # Only pool if spatial dims > 1 (avoid collapsing to 0)
        if i < 4:
            model.add(MaxPooling2D(POOL_SIZE))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def build_wide_cnn(loss: str = "sparse_categorical_crossentropy") -> Sequential:
    """Model 7 — Wide CNN: fewer layers but many more filters per layer.

    Architecture: Conv(128) → Pool → Conv(256) → Pool → Dense(256) → Dense(10)

    Like a huge team of specialists all on the same level — lots of
    eyes but only one round of review. Trades depth for width.
    Expected accuracy: ~89-91%.
    """
    model = Sequential([
        Conv2D(128, KERNEL_SIZE, activation="relu", input_shape=CNN_INPUT_SHAPE),
        MaxPooling2D(POOL_SIZE),
        Conv2D(256, KERNEL_SIZE, activation="relu"),
        MaxPooling2D(POOL_SIZE),
        Flatten(),
        Dense(256, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax"),
    ], name="model_07_wide_cnn")

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model
