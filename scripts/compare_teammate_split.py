"""Head-to-head comparison against a teammate's network, using the teammate's
splitting procedure instead of ours so both models see identical rows.

Their protocol, reproduced exactly (seed 0 throughout):
  1. outer_holdout_split: 20% of X_trn held out as a labelled stand-in for X_test.
  2. train_val_split on the remaining pool: 20% of it for early stopping.
  3. Train on the rest, score on the holdout.

Their splits are plain random permutations (not stratified like ours), so the
default rate differs slightly between parts; this is printed below.

Our preprocessing and hyperparameters are unchanged (from _common), and nothing
is saved to artifacts/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml_assignment.data import Preprocessor, load_raw
from ml_assignment.losses import brier_score, economic_loss
from ml_assignment.mlp import predict
from ml_assignment.train import train_mlp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import HYPERPARAMS, PREPROCESS

TEAMMATE_SEED = 0


# ---- teammate's splitting code, copied verbatim ----------------------------

def train_val_split(
    X: pd.DataFrame, y: pd.Series, val_frac: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random train/validation split. Returns (X_tr, X_val, y_tr, y_val) as
    plain numpy arrays, with val_frac of the rows going to validation."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, trn_idx = idx[:n_val], idx[n_val:]
    return (
        X.iloc[trn_idx].to_numpy(),
        X.iloc[val_idx].to_numpy(),
        y.iloc[trn_idx].to_numpy(),
        y.iloc[val_idx].to_numpy(),
    )


def outer_holdout_split(
    X: pd.DataFrame, y: pd.Series, holdout_frac: float = 0.2, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split off a held-out chunk with known labels, standing in for X_test.
    Returns (X_pool, X_holdout, y_pool, y_holdout)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_holdout = int(n * holdout_frac)
    holdout_idx, pool_idx = idx[:n_holdout], idx[n_holdout:]
    return X.iloc[pool_idx], X.iloc[holdout_idx], y.iloc[pool_idx], y.iloc[holdout_idx]


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X_trn, y_trn, _ = load_raw()

    X_pool, X_hold, y_pool, y_hold = outer_holdout_split(
        X_trn, y_trn, holdout_frac=0.2, seed=TEAMMATE_SEED
    )
    X_tr, X_val, y_tr, y_val = train_val_split(
        X_pool, y_pool, val_frac=0.2, seed=TEAMMATE_SEED
    )
    # Their train_val_split drops column names; our Preprocessor selects by name.
    X_tr = pd.DataFrame(X_tr, columns=X_trn.columns)
    X_val = pd.DataFrame(X_val, columns=X_trn.columns)
    y_tr, y_val = y_tr.astype(float), y_val.astype(float)
    y_hold = y_hold.to_numpy().astype(float)

    print(f"split sizes: train {len(y_tr)} | val {len(y_val)} | holdout {len(y_hold)}")
    print(f"default rate: train {y_tr.mean():.4f} | val {y_val.mean():.4f} | "
          f"holdout {y_hold.mean():.4f}\n")

    pp = Preprocessor(**PREPROCESS).fit(X_tr)
    Z_tr, Z_val, Z_hold = pp.transform(X_tr), pp.transform(X_val), pp.transform(X_hold)

    rows = {}
    for name, w in (("Brier-trained", 1.0), ("Economic-trained", 3.0)):
        res = train_mlp(Z_tr, y_tr, Z_val, y_val, w_default=w, **HYPERPARAMS)
        p = predict(res["params"], Z_hold)
        rows[name] = [brier_score(y_hold, p), economic_loss(y_hold, p), p.mean()]
        print(f"[{name}] stopped at epoch {res['best_epoch']} of {res['epochs_run']}")

    table = pd.DataFrame(rows, index=["Brier score", "Economic loss", "mean p"]).T
    print(f"\n=== holdout performance (teammate split, seed {TEAMMATE_SEED}, "
          f"n={len(y_hold)}) ===")
    print(table.round(5).to_string())
