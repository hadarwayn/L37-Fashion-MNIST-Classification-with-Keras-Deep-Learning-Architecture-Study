"""
Data loader — loads Fashion-MNIST and prepares it for training.

Fashion-MNIST is built into Keras, so no downloads needed!
It has 70,000 grayscale images (28×28 pixels) of clothing items
split into 10 categories like T-shirt, Trouser, Sneaker, etc.
"""

import numpy as np
from typing import Dict, Any

# The 10 clothing categories (index = label number)
CLASS_NAMES: list[str] = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

RANDOM_SEED = 42
NUM_CLASSES = 10
IMG_HEIGHT = 28
IMG_WIDTH = 28
PIXEL_MAX = 255.0
TRAIN_SIZE = 50000
VAL_SIZE = 10000
MIN_CLASS_THRESHOLD = 4000
MAX_CLASS_THRESHOLD = 7000


def load_fashion_mnist() -> Dict[str, Any]:
    """Load Fashion-MNIST, normalize, split, and reshape for both FC and CNN.

    Returns:
        Dictionary with all data splits and metadata:
        - x_train, y_train         (50K, for training)
        - x_val, y_val             (10K, for validation during training)
        - x_test, y_test           (10K, final evaluation — never seen)
        - x_train_cnn, x_val_cnn, x_test_cnn   (reshaped with channel dim)
        - y_train_onehot, y_val_onehot, y_test_onehot  (one-hot encoded)
        - class_names, num_classes
    """
    from tensorflow.keras.datasets import fashion_mnist
    from tensorflow.keras.utils import to_categorical

    np.random.seed(RANDOM_SEED)

    # Step 1: Load raw data (Keras gives us 60K train + 10K test)
    (x_full_train, y_full_train), (x_test, y_test) = fashion_mnist.load_data()

    # Step 2: Normalize pixel values from 0–255 → 0.0–1.0
    x_full_train = x_full_train.astype("float32") / PIXEL_MAX
    x_test = x_test.astype("float32") / PIXEL_MAX

    # Step 3: Split 60K into 50K train + 10K validation
    indices = np.random.permutation(len(x_full_train))
    train_idx = indices[:TRAIN_SIZE]
    val_idx = indices[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]

    x_train = x_full_train[train_idx]
    y_train = y_full_train[train_idx]
    x_val = x_full_train[val_idx]
    y_val = y_full_train[val_idx]

    # Step 4: Reshape for CNN — add channel dimension (28,28) → (28,28,1)
    x_train_cnn = x_train[..., np.newaxis]
    x_val_cnn = x_val[..., np.newaxis]
    x_test_cnn = x_test[..., np.newaxis]

    # Step 5: One-hot encode labels for categorical crossentropy
    y_train_onehot = to_categorical(y_train, NUM_CLASSES)
    y_val_onehot = to_categorical(y_val, NUM_CLASSES)
    y_test_onehot = to_categorical(y_test, NUM_CLASSES)

    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
        "x_train_cnn": x_train_cnn,
        "x_val_cnn": x_val_cnn,
        "x_test_cnn": x_test_cnn,
        "y_train_onehot": y_train_onehot,
        "y_val_onehot": y_val_onehot,
        "y_test_onehot": y_test_onehot,
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
    }


def print_class_distribution(data: Dict[str, Any]) -> None:
    """Print a table showing how many images per class in each split.

    This is MANDATORY before training — we need balanced classes.
    If 90% of data were T-shirts, the model might just always guess
    'T-shirt' and look accurate without learning anything.
    """
    splits = [
        ("Train", data["y_train"]),
        ("Validation", data["y_val"]),
        ("Test", data["y_test"]),
    ]
    print(f"\n{'Class':<16} ", end="")
    for name, _ in splits:
        print(f"{'':>2}{name:>8}", end="")
    print()
    print("-" * 46)

    for i, cname in enumerate(CLASS_NAMES):
        print(f"{cname:<16} ", end="")
        for _, labels in splits:
            count = int(np.sum(labels == i))
            print(f"{count:>10}", end="")
        print()

    # Warn about imbalance
    train_counts = [int(np.sum(data["y_train"] == i)) for i in range(NUM_CLASSES)]
    min_c, max_c = min(train_counts), max(train_counts)
    if min_c < MIN_CLASS_THRESHOLD or max_c > MAX_CLASS_THRESHOLD:
        print("\n[!] WARNING: Significant class imbalance detected!")
    else:
        print("\n[OK] Classes are well-balanced -- safe to train.")
