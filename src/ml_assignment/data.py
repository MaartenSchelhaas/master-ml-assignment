"""Loading and train/validation splitting. All preprocessing fit only on
the training portion, then applied unchanged elsewhere."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_raw() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load X_trn, y_trn, X_test from data/, as given on Canvas."""
    X_trn = pd.read_csv(DATA_DIR / "X_trn.csv")
    y_trn = pd.read_csv(DATA_DIR / "y_trn.csv").iloc[:, 0]
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    return X_trn, y_trn, X_test


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


def k_fold_split(
    X: pd.DataFrame, y: pd.Series, k: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Split X, y into k folds for cross-validation.

    Args:
        X (pd.DataFrame): Features, shape (n, n_features).
        y (pd.Series): Labels, shape (n,).
        k (int, optional): Number of folds. Defaults to 5.
        seed (int, optional): Random seed for the fold assignment. Defaults
            to 0.

    Returns:
        list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]: k
            entries of (X_tr, X_val, y_tr, y_val), one per held-out fold,
            same array layout as train_val_split.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)

    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_parts = []
        for j in range(k):
            if j != i:
                train_parts.append(folds[j])
        trn_idx = np.concatenate(train_parts)

        splits.append((
            X.iloc[trn_idx].to_numpy(),
            X.iloc[val_idx].to_numpy(),
            y.iloc[trn_idx].to_numpy(),
            y.iloc[val_idx].to_numpy(),
        ))
    return splits


def fit_standardizer(X_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and standard deviation from training data.

    Args:
        X_tr (np.ndarray): Training features, shape (n_train, n_features).

    Returns:
        tuple[np.ndarray, np.ndarray]: (mean, std), each shape
            (n_features,), to be passed into standardize for X_tr, X_val
            and X_test alike.
    """
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    return mean, std


def standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply z-score standardization: (X - mean) / std, elementwise per
    feature.

    Args:
        X (np.ndarray): Features to standardize, shape (n, n_features).
        mean (np.ndarray): Per-feature mean, as returned by
            fit_standardizer.
        std (np.ndarray): Per-feature standard deviation, as returned by
            fit_standardizer.

    Returns:
        np.ndarray: Standardized features, same shape as X.
    """
    return (X - mean) / std
