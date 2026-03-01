"""
Unified training pipeline — train any model, measure time, collect metrics.

Handles the full lifecycle: build → train → evaluate → record results.
Works with all 10 architectures and both loss function types.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional

from sklearn.metrics import confusion_matrix
from src.models import get_model, MODEL_INFO
from src.utils.logger import setup_logger

logger = setup_logger("trainer")

EARLY_STOP_PATIENCE = 5
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 64


def train_model(
    model,
    data: Dict[str, Any],
    model_id: int,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    use_onehot: bool = False,
) -> Dict[str, Any]:
    """Train one model and return metrics + history.

    Args:
        model: Compiled Keras model.
        data: Dict from load_fashion_mnist().
        model_id: Integer 1–10.
        epochs: Max training epochs.
        batch_size: Samples per gradient step.
        use_onehot: If True, use one-hot labels (for categorical CE).

    Returns:
        Dict with history, test metrics, predictions, timing, etc.
    """
    from tensorflow.keras.callbacks import EarlyStopping

    info = MODEL_INFO.get(model_id, {})
    name = info.get("name", f"Model {model_id}")
    mtype = info.get("type", "CNN")

    # Pick correct data format based on model type and loss
    x_train, x_val, x_test = _pick_inputs(data, mtype)
    y_train = data["y_train_onehot"] if use_onehot else data["y_train"]
    y_val = data["y_val_onehot"] if use_onehot else data["y_val"]
    y_test = data["y_test_onehot"] if use_onehot else data["y_test"]

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
    )

    logger.info(f"Training {name} (ID={model_id}) for up to {epochs} epochs...")
    print(f"\n{'='*55}")
    print(f"  Training Model {model_id}/10: {name}")
    print(f"{'='*55}")

    start = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )
    elapsed = time.time() - start

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    # Predictions for confusion matrix
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1) if use_onehot else y_test
    cm = confusion_matrix(y_true, y_pred)

    total_params = model.count_params()
    epochs_trained = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    result = {
        "model_id": model_id,
        "model_name": name,
        "group": info.get("group", "?"),
        "type": mtype,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "train_accuracy": float(history.history["accuracy"][-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1]),
        "training_time_seconds": round(elapsed, 1),
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "total_params": total_params,
        "history": history.history,
        "y_pred": y_pred,
        "y_true": y_true,
        "confusion_matrix": cm,
    }

    logger.info(f"{name}: test_acc={test_acc:.4f}, time={elapsed:.1f}s")
    print(f"  Result: {test_acc:.4f} accuracy in {elapsed:.1f}s "
          f"({epochs_trained} epochs, best@{best_epoch})\n")
    return result


def train_all_models(
    model_ids: List[int],
    data: Dict[str, Any],
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """Train multiple models sequentially and collect all results."""
    results = []
    total = len(model_ids)
    for i, mid in enumerate(model_ids, 1):
        print(f"\n[{i}/{total}] Building model {mid}...")
        try:
            model = get_model(mid)
            r = train_model(model, data, mid, epochs, batch_size)
            results.append(r)
        except Exception as e:
            logger.error(f"Model {mid} failed: {e}")
            print(f"  ERROR: Model {mid} failed — {e}. Continuing...\n")
    return results


def _pick_inputs(data: Dict, model_type: str):
    """Return the right x arrays based on model type (FC vs CNN)."""
    if model_type == "FC":
        return data["x_train"], data["x_val"], data["x_test"]
    return data["x_train_cnn"], data["x_val_cnn"], data["x_test_cnn"]
