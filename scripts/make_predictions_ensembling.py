"""Produce predictions.npy the same way as make_predictions.py, but each
final model is an ensemble: fit n_ensemble times on independently resampled
train/val splits (same hyperparameters every time, only the seed varies),
then average the n_ensemble predictions. Reduces variance from any one
arbitrary split/initialization, on top of the hyperparameters already
picked by scripts/tune.py.

The real X_test has no labels, so we can't score the ensemble's own
aggregated prediction against ground truth directly. To still get an honest
read on that, we run the exact same procedure twice: once on a held-out
chunk of X_trn that stands in for X_test (so we can score it, for the
report), and once for real on the full X_trn -> the actual X_test (for
predictions.npy, unscored).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ml_assignment.data import derive_seeds, fit_standardizer, load_raw, outer_holdout_split, standardize, train_val_split
from ml_assignment.losses import BrierLoss, EconomicLoss, Loss
from ml_assignment.model import MLP
from make_predictions import make_optimizer, outer_holdout_frac, outer_seed, report

# hyperparameters picked by scripts/tune.py, one config per final model
brier_config = {"hidden_sizes": [16, 8], "lr": 0.01, "batch_size": 32, "optimizer": "adam"}
econ_config = {"hidden_sizes": [8], "lr": 0.01, "batch_size": 128, "optimizer": "adam"}

n_ensemble = 20
master_seed = 0
n_epochs = 300
patience = 30

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


def fit_one_member(config: dict, loss: Loss, X_pool: pd.DataFrame, y_pool: pd.Series, X_target: pd.DataFrame, seed: int) -> np.ndarray:
    """Fit one ensemble member on its own resampled train/val split, drawn
    from X_pool/y_pool, then predict on X_target.

    Args:
        config (dict): Fixed hyperparameters, shared by every member.
        loss (Loss): Loss to train with.
        X_pool (pd.DataFrame): Pool to resample this member's own
            train/val split from.
        y_pool (pd.Series): Labels matching X_pool.
        X_target (pd.DataFrame): Features to predict on, standardized with
            this member's own X_tr statistics first.
        seed (int): Used for both the train/val split and the model's own
            initialization/shuffling.

    Returns:
        np.ndarray: This member's predicted probabilities for X_target.
    """
    X_tr, X_val, y_tr, y_val = train_val_split(X_pool, y_pool, seed=seed)
    mean, std = fit_standardizer(X_tr)
    X_tr = standardize(X_tr, mean, std)
    X_val = standardize(X_val, mean, std)
    X_target_arr = standardize(X_target.to_numpy(), mean, std)

    model = MLP(
        input_dim=X_tr.shape[1],
        hidden_sizes=config["hidden_sizes"],
        optimizer=make_optimizer(config["optimizer"], config["lr"]),
        batch_size=config["batch_size"],
        n_epochs=n_epochs,
        loss=loss,
        patience=patience,
        seed=seed,
    )
    model.fit(X_tr, y_tr, X_val, y_val)
    return model.predict_proba(X_target_arr)


def ensemble_predict(
    config: dict, loss: Loss, X_pool: pd.DataFrame, y_pool: pd.Series, X_target: pd.DataFrame, seeds: list[int]
) -> np.ndarray:
    """Fit one member per seed, average their X_target predictions."""
    p_members = []
    for seed in seeds:
        p_members.append(fit_one_member(config, loss, X_pool, y_pool, X_target, seed))
    return np.mean(p_members, axis=0)


if __name__ == "__main__":
    X_trn, y_trn, X_test = load_raw()
    OUTPUT_DIR.mkdir(exist_ok=True)

    seeds = derive_seeds(master_seed, n_ensemble)

    # Report: run the exact same procedure on a held-out chunk of X_trn
    # that stands in for X_test, so the aggregated ensemble prediction can
    # actually be scored against known labels.
    X_pool, X_holdout, y_pool, y_holdout = outer_holdout_split(X_trn, y_trn, outer_holdout_frac, outer_seed)
    p_holdout_brier = ensemble_predict(brier_config, BrierLoss(), X_pool, y_pool, X_holdout, seeds)
    p_holdout_econ = ensemble_predict(econ_config, EconomicLoss(), X_pool, y_pool, X_holdout, seeds)

    y_holdout_arr = y_holdout.to_numpy()
    metrics_text = report("Brier model", y_holdout_arr, p_holdout_brier)
    metrics_text += "\n" + report("Economic model", y_holdout_arr, p_holdout_econ)
    (OUTPUT_DIR / "metrics_ensemble.txt").write_text(metrics_text)

    # Real deliverable: same procedure on the full X_trn (no holdout carved
    # out, X_test is already a separate, untouched set) -> predict on the
    # real X_test, which has no labels to score against.
    p_brier = ensemble_predict(brier_config, BrierLoss(), X_trn, y_trn, X_test, seeds)
    p_econ = ensemble_predict(econ_config, EconomicLoss(), X_trn, y_trn, X_test, seeds)

    predictions = np.column_stack([p_brier, p_econ])
    assert predictions.shape == (len(X_test), 2)
    assert np.all(np.isfinite(predictions))
    assert np.all((predictions >= 0) & (predictions <= 1))

    np.save(OUTPUT_DIR / "predictions_ensemble.npy", predictions)

    print(f"saved predictions_ensemble.npy with shape {predictions.shape} to {OUTPUT_DIR}")
    print(metrics_text)
