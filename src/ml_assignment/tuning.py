"""Cross-validated hyperparameter search.

Two properties matter here. First, the preprocessor is refitted inside every
fold, on that fold's training part only, so the search never lets validation
rows influence a transformation. Second, each objective is tuned under its own
loss: the configuration that is best for the Brier score need not be best for
the economic loss, and we are graded on both separately.

The search space below is the documented list of candidate values the report
has to report; it was narrowed by the one-factor probe in the scratchpad, which
showed L2 selecting at the edge of the originally planned range and smaller
networks beating larger ones.
"""

from itertools import product

import numpy as np
import pandas as pd

from .data import Preprocessor
from .mlp import predict
from .train import train_mlp

SEARCH_SPACE = {
    "hidden": [(16,), (32,), (64,), (32, 16), (64, 32)],
    "lr": [3e-4, 1e-3, 3e-3],
    "l2": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "one_hot": [True, False],
}
BATCH_SIZE = 256      # probe showed no measurable effect; fixed to save budget
MAX_EPOCHS = 300
PATIENCE = 20


def stratified_kfold(
    y: np.ndarray, n_splits: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fold indices with the default rate held constant across folds.

    Each class is shuffled and dealt round-robin into folds, so every fold sees
    the same 22% base rate. This matters most for the economic loss, which
    triples the influence of the default observations.
    """
    rng = np.random.default_rng(seed)
    fold_id = np.empty(len(y), dtype=int)
    for label in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == label))
        fold_id[idx] = np.arange(len(idx)) % n_splits
    return [
        (np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k))
        for k in range(n_splits)
    ]


def sample_configs(n_configs: int, seed: int = 0) -> list[dict]:
    """Random search: draw without replacement from the full grid."""
    keys = list(SEARCH_SPACE)
    grid = [dict(zip(keys, vals)) for vals in product(*(SEARCH_SPACE[k] for k in keys))]
    rng = np.random.default_rng(seed)
    return [grid[i] for i in rng.permutation(len(grid))[:n_configs]]


def cv_score(
    X: pd.DataFrame,
    y: np.ndarray,
    config: dict,
    *,
    w_default: float,
    n_splits: int = 5,
    split_seed: int = 0,
) -> dict:
    """Mean out-of-fold loss for one configuration, under its own objective."""
    losses, epochs = [], []
    for fold, (tr_idx, va_idx) in enumerate(stratified_kfold(y, n_splits, split_seed)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        pp = Preprocessor(one_hot=config["one_hot"]).fit(X_tr)   # fit inside the fold
        res = train_mlp(
            pp.transform(X_tr), y[tr_idx],
            pp.transform(X_va), y[va_idx],
            hidden=config["hidden"], lr=config["lr"], l2=config["l2"],
            batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
            w_default=w_default, seed=fold,
        )
        losses.append(res["val_loss"])
        epochs.append(res["best_epoch"])
    return {
        "cv_loss": float(np.mean(losses)),
        "cv_sd": float(np.std(losses)),
        "mean_best_epoch": float(np.mean(epochs)),
        "fold_losses": losses,
    }


def run_search(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    w_default: float,
    n_configs: int = 30,
    n_splits: int = 5,
    search_seed: int = 0,
    split_seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Random search, returned as a table sorted best-first."""
    rows = []
    configs = sample_configs(n_configs, seed=search_seed)
    for i, config in enumerate(configs, 1):
        out = cv_score(X, y, config, w_default=w_default,
                       n_splits=n_splits, split_seed=split_seed)
        rows.append({
            "hidden": str(config["hidden"]), "lr": config["lr"], "l2": config["l2"],
            "one_hot": config["one_hot"], "batch_size": BATCH_SIZE,
            "cv_loss": out["cv_loss"], "cv_sd": out["cv_sd"],
            "mean_best_epoch": out["mean_best_epoch"],
        })
        if verbose:
            print(f"  [{i:2d}/{len(configs)}] {str(config['hidden']):<9} "
                  f"lr={config['lr']:<7g} l2={config['l2']:<7g} "
                  f"oh={str(config['one_hot']):<5} -> "
                  f"cv={out['cv_loss']:.5f} (sd {out['cv_sd']:.5f}, "
                  f"ep {out['mean_best_epoch']:.0f})", flush=True)
    return pd.DataFrame(rows).sort_values("cv_loss").reset_index(drop=True)


def oof_predictions(
    X: pd.DataFrame,
    y: np.ndarray,
    config: dict,
    *,
    w_default: float,
    n_splits: int = 5,
    split_seed: int = 0,
) -> tuple[np.ndarray, list[dict]]:
    """Out-of-fold predictions for every training row.

    Each row is predicted by a model that never saw it, so the resulting vector
    can be scored under either loss to fill the report's 2x2 table on all 25,000
    observations rather than on a single 5,000-row split.

    Early stopping is nested: the outer fold's training part is split again, and
    the stopping epoch is chosen on that inner slice. The outer fold is therefore
    untouched by training, preprocessing and the stopping rule alike -- unlike
    the step 5 search, where the scored fold also chose the epoch.
    """
    p_oof = np.full(len(y), np.nan)
    info = []
    for fold, (tr_idx, va_idx) in enumerate(stratified_kfold(y, n_splits, split_seed)):
        X_out_tr, X_out_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_out_tr = y[tr_idx]

        # Preprocessing is fitted on the outer training part only.
        pp = Preprocessor(one_hot=config["one_hot"]).fit(X_out_tr)
        Z_out_tr = pp.transform(X_out_tr)

        # Inner 80/20 split of that part, used only to pick the stopping epoch.
        in_tr, in_va = stratified_kfold(y_out_tr, n_splits=5, seed=split_seed + 100)[0]

        res = train_mlp(
            Z_out_tr[in_tr], y_out_tr[in_tr],
            Z_out_tr[in_va], y_out_tr[in_va],
            hidden=config["hidden"], lr=config["lr"], l2=config["l2"],
            batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
            w_default=w_default, seed=fold,
        )
        p_oof[va_idx] = predict(res["params"], pp.transform(X_out_va))
        info.append({
            "fold": fold,
            "best_epoch": res["best_epoch"],
            "n_val": len(va_idx),
            "inner_val_loss": res["val_loss"],
        })

    assert np.isfinite(p_oof).all(), "every row must receive an out-of-fold prediction"
    return p_oof, info
