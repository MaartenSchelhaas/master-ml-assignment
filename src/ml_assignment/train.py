"""Shared training loop for both final models.

The Brier model and the economic model differ in exactly one argument,
`w_default`: 1.0 reproduces the Brier score, 3.0 the asymmetric economic loss.
Everything else (architecture, preprocessing, optimizer, stopping rule) is held
fixed, which is what makes the two-model comparison in the report interpretable.
"""

from pathlib import Path

import numpy as np

from .losses import weighted_mse, weights_for
from .mlp import (
    adam_update,
    backward,
    forward,
    init_adam_state,
    init_params,
    predict,
)


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _copy_params(params: list[dict]) -> list[dict]:
    return [{k: v.copy() for k, v in layer.items()} for layer in params]


def train_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden: tuple[int, ...] = (64, 32),
    w_default: float = 1.0,
    lr: float = 1e-3,
    l2: float = 0.0,
    batch_size: int = 256,
    max_epochs: int = 200,
    patience: int = 15,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Train one network by mini-batch Adam, with early stopping.

    Inputs are already preprocessed arrays: `train_mlp` never sees raw columns,
    so it cannot leak anything across the split.

    Early stopping watches validation loss *under the training objective*, so
    each model is stopped by the criterion it is being graded on. The returned
    parameters are those from the best epoch, not the last one.
    """
    n, d = X_tr.shape
    rng = np.random.default_rng(seed)

    base_rate = float(np.mean(y_tr))
    params = init_params([d, *hidden, 1], seed=seed, output_bias=_logit(base_rate))
    state = init_adam_state(params)

    w_tr = weights_for(y_tr, w_default)
    w_val = weights_for(y_val, w_default)

    best_loss, best_epoch, best_params = np.inf, -1, _copy_params(params)
    history = {"train": [], "val": []}
    stale = 0

    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            _, cache = forward(params, X_tr[idx])
            grads = backward(params, cache, y_tr[idx], w_tr[idx], l2=l2)
            adam_update(params, grads, state, lr=lr)

        tr_loss = weighted_mse(y_tr, predict(params, X_tr), w_tr)
        val_loss = weighted_mse(y_val, predict(params, X_val), w_val)
        history["train"].append(tr_loss)
        history["val"].append(val_loss)

        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, best_params = val_loss, epoch, _copy_params(params)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  train {tr_loss:.5f}  val {val_loss:.5f}")

    return {
        "params": best_params,
        "best_epoch": best_epoch,
        "val_loss": best_loss,
        "history": history,
        "epochs_run": len(history["train"]),
        "config": {
            "hidden": tuple(hidden),
            "w_default": w_default,
            "lr": lr,
            "l2": l2,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "seed": seed,
            "n_features": d,
        },
    }


def save_model(path: str | Path, params: list[dict], config: dict) -> None:
    """Store weights flat (W0, b0, W1, ...) alongside the config, so a run can
    be reproduced and the predictions regenerated without retraining."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for i, layer in enumerate(params):
        flat[f"W{i}"] = layer["W"]
        flat[f"b{i}"] = layer["b"]
    np.savez(path, n_layers=len(params), config=np.array(repr(config)), **flat)


def load_model(path: str | Path) -> list[dict]:
    with np.load(path, allow_pickle=False) as f:
        return [
            {"W": f[f"W{i}"], "b": f[f"b{i}"]} for i in range(int(f["n_layers"]))
        ]


def train_full(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    hidden: tuple[int, ...] = (64,),
    w_default: float = 1.0,
    lr: float = 1e-3,
    l2: float = 0.0,
    batch_size: int = 256,
    seed: int = 0,
) -> list[dict]:
    """Refit on the full training set for a fixed number of epochs.

    The final models are trained on all labelled data, which leaves no
    validation set to stop on. The epoch count therefore comes from
    cross-validation -- the median best epoch across folds -- rather than from
    early stopping. Note the CV models saw fewer rows per epoch than this refit
    does, so the same epoch count means somewhat more gradient steps here; that
    is the conservative direction, since the extra data regularizes.
    """
    n, d = X.shape
    rng = np.random.default_rng(seed)
    params = init_params([d, *hidden, 1], seed=seed, output_bias=_logit(float(np.mean(y))))
    state = init_adam_state(params)
    w = weights_for(y, w_default)

    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            _, cache = forward(params, X[idx])
            adam_update(params, backward(params, cache, y[idx], w[idx], l2=l2), state, lr=lr)
    return params
