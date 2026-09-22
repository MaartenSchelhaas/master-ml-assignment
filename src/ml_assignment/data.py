"""Loading and train/validation splitting. All preprocessing fit only on
the training portion, then applied unchanged elsewhere."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_raw() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X_trn = pd.read_csv(DATA_DIR / "X_trn.csv")
    y_trn = pd.read_csv(DATA_DIR / "y_trn.csv").squeeze("columns")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    return X_trn, y_trn, X_test


def train_val_split(
    X: pd.DataFrame, y: pd.Series, val_frac: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
