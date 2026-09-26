"""Produce predictions.npy: shape (n_test, 2), column 0 = Brier model,
column 1 = economic-loss model, same row order as X_test.csv. Also writes
each model's validation confusion matrix and accuracy to output/, as a
sanity check (the test set has no labels to score against)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ml_assignment.data import fit_standardizer, load_raw, standardize, train_val_split
from ml_assignment.losses import BrierLoss, EconomicLoss
from ml_assignment.model import MLP
from ml_assignment.optim import SGD

# hyperparameters picked by scripts/tune.py
hidden_sizes = [8]
lr = 0.1
batch_size = 32

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """2x2 confusion matrix for binary 0/1 labels.

    Args:
        y_true (np.ndarray): True labels, shape (n,).
        y_pred (np.ndarray): Predicted labels, shape (n,).

    Returns:
        np.ndarray: [[tn, fp], [fn, tp]].
    """
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of labels predicted correctly."""
    return float(np.mean(y_true == y_pred))


def plot_histories(histories: dict[str, dict[str, list[float]]], path: Path) -> None:
    """Plot train_loss/val_loss per epoch, one subplot per model.

    Args:
        histories (dict[str, dict[str, list[float]]]): Model name ->
            history dict, as returned by MLP.fit.
        path (Path): Where to save the figure.
    """
    n_models = len(histories)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4))

    i = 0
    for name, history in histories.items():
        ax = axes[i]
        ax.plot(history["train_loss"], label="train_loss")
        ax.plot(history["val_loss"], label="val_loss")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        i += 1

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def report(name: str, y_val: np.ndarray, p_val: np.ndarray, loss_value: float) -> str:
    """Build the loss / confusion matrix / accuracy text block for one model."""
    y_pred = (p_val >= 0.5).astype(int)
    cm = confusion_matrix(y_val, y_pred)
    acc = accuracy(y_val, y_pred)
    return (
        f"{name}\n"
        f"loss: {loss_value:.4f}\n"
        f"confusion matrix [[tn, fp], [fn, tp]]:\n{cm}\n"
        f"accuracy: {acc:.4f}\n"
    )


if __name__ == "__main__":
    X_trn, y_trn, X_test = load_raw()
    X_tr, X_val, y_tr, y_val = train_val_split(X_trn, y_trn)
    OUTPUT_DIR.mkdir(exist_ok=True)

    mean, std = fit_standardizer(X_tr)
    X_tr = standardize(X_tr, mean, std)
    X_val = standardize(X_val, mean, std)
    X_test_arr = standardize(X_test.to_numpy(), mean, std)

    brier_model = MLP(
        input_dim=X_tr.shape[1],
        hidden_sizes=hidden_sizes,
        optimizer=SGD(lr=lr),
        batch_size=batch_size,
        n_epochs=300,
        loss=BrierLoss(),
        patience=30,
        seed=0,
    )
    history_brier = brier_model.fit(X_tr, y_tr, X_val, y_val)
    p_brier = brier_model.predict_proba(X_test_arr)

    econ_model = MLP(
        input_dim=X_tr.shape[1],
        hidden_sizes=hidden_sizes,
        optimizer=SGD(lr=lr),
        batch_size=batch_size,
        n_epochs=300,
        loss=EconomicLoss(),
        patience=30,
        seed=0,
    )
    history_econ = econ_model.fit(X_tr, y_tr, X_val, y_val)
    p_econ = econ_model.predict_proba(X_test_arr)

    predictions = np.column_stack([p_brier, p_econ])
    assert predictions.shape == (len(X_test), 2)
    assert np.all(np.isfinite(predictions))
    assert np.all((predictions >= 0) & (predictions <= 1))

    np.save(OUTPUT_DIR / "predictions.npy", predictions)
    plot_histories({"Brier model": history_brier, "Economic model": history_econ}, OUTPUT_DIR / "history.png")

    p_val_brier = brier_model.predict_proba(X_val)
    p_val_econ = econ_model.predict_proba(X_val)
    print(f"p_val_brier: min={p_val_brier.min():.4f} mean={p_val_brier.mean():.4f} max={p_val_brier.max():.4f}")
    print(f"p_val_econ:  min={p_val_econ.min():.4f} mean={p_val_econ.mean():.4f} max={p_val_econ.max():.4f}")

    metrics_text = report("Brier model", y_val, p_val_brier, brier_model.loss.value(y_val, p_val_brier))
    metrics_text += "\n" + report("Economic model", y_val, p_val_econ, econ_model.loss.value(y_val, p_val_econ))
    (OUTPUT_DIR / "metrics.txt").write_text(metrics_text)

    print(f"saved predictions.npy with shape {predictions.shape} to {OUTPUT_DIR}")
    print(metrics_text)