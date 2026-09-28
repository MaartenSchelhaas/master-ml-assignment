"""Loading, splitting and preprocessing.

Every transformation is fitted on the training portion only and then applied
unchanged to validation and test observations, as the assignment requires. The
`Preprocessor` below is the only place that learns anything from the data, and
it learns it in `fit`; `transform` never looks at the data it is given beyond
applying stored constants.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Column groups, read off the training data (see the profiling in the README).
# The monetary columns are heavily right-skewed amounts that also take negative
# values and exact zeros, so they get a signed log. The ordinal columns are the
# repayment-status style variables: roughly ten ordered levels each, carrying
# most of the univariate signal, with very sparse extreme levels.
MONETARY_COLS = [
    "X01", "X02", "X08", "X09", "X10", "X11",
    "X12", "X14", "X18", "X19", "X20", "X21", "X23",
]
ORDINAL_COLS = ["X03", "X04", "X06", "X07", "X13", "X15", "X16", "X17", "X22"]
CONTINUOUS_COLS = ["X05"]
ALL_COLS = MONETARY_COLS + ORDINAL_COLS + CONTINUOUS_COLS


def load_raw() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X_trn = pd.read_csv(DATA_DIR / "X_trn.csv")
    y_trn = pd.read_csv(DATA_DIR / "Y_trn.csv").squeeze("columns")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    return X_trn, y_trn, X_test


def train_val_split(
    X: pd.DataFrame, y: pd.Series, val_frac: float = 0.2, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Stratified split, so both parts carry the same default rate.

    Stratifying matters at a 22% base rate: an unstratified 20% validation split
    has a noticeably variable number of defaults, which adds noise to the
    economic loss in particular, since it weights those observations by three.

    X comes back as a DataFrame because `Preprocessor` works by column name;
    y comes back as a float array because the losses and gradients want arrays.
    """
    rng = np.random.default_rng(seed)
    y_arr = y.to_numpy()
    val_mask = np.zeros(len(X), dtype=bool)
    for label in (0, 1):
        idx = np.flatnonzero(y_arr == label)
        idx = rng.permutation(idx)
        val_mask[idx[: int(round(len(idx) * val_frac))]] = True
    return (
        X.loc[~val_mask],
        X.loc[val_mask],
        y_arr[~val_mask].astype(float),
        y_arr[val_mask].astype(float),
    )


def signed_log1p(a: np.ndarray) -> np.ndarray:
    """sign(x) * log1p(|x|): compresses the long right tail while keeping the
    sign of negative balances and mapping exact zeros to zero."""
    return np.sign(a) * np.log1p(np.abs(a))


class Preprocessor:
    """Fit on training data, apply unchanged everywhere else.

    Pipeline: clip to the range learned from training, signed-log the monetary
    columns, standardize every column, and optionally append one-hot indicators
    for the dense levels of the ordinal columns.
    """

    def __init__(
        self,
        log_monetary: bool = True,
        winsor_q: float = 0.001,
        one_hot: bool = False,
        min_level_count: int = 100,
    ):
        self.log_monetary = log_monetary
        self.winsor_q = winsor_q
        self.one_hot = one_hot
        self.min_level_count = min_level_count

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        X = X[ALL_COLS]
        self.columns_ = list(X.columns)

        # Clipping bounds. Monetary and continuous columns are winsorized at
        # training quantiles to stop a handful of extreme values dominating the
        # scale. Ordinal columns are clipped only to the observed training range,
        # since their levels are meaningful and sparsity is handled by one-hot.
        lower, upper = {}, {}
        for col in self.columns_:
            s = X[col]
            if col in ORDINAL_COLS:
                lower[col], upper[col] = float(s.min()), float(s.max())
            else:
                lower[col] = float(s.quantile(self.winsor_q))
                upper[col] = float(s.quantile(1.0 - self.winsor_q))
        self.lower_ = np.array([lower[c] for c in self.columns_])
        self.upper_ = np.array([upper[c] for c in self.columns_])

        # Levels kept as indicators, learned from training counts only.
        self.levels_ = {}
        if self.one_hot:
            for col in ORDINAL_COLS:
                counts = X[col].value_counts()
                self.levels_[col] = np.sort(
                    counts[counts >= self.min_level_count].index.to_numpy()
                )

        # Standardization constants, computed after clipping and logging so they
        # describe the values the network actually sees.
        Z = self._numeric_block(X)
        self.mean_ = Z.mean(axis=0)
        std = Z.std(axis=0)
        self.std_ = np.where(std < 1e-12, 1.0, std)
        return self

    def _numeric_block(self, X: pd.DataFrame) -> np.ndarray:
        """Clip and signed-log, without standardizing. Shared by fit and
        transform so both see identical intermediate values."""
        Z = X[self.columns_].to_numpy(dtype=float)
        Z = np.clip(Z, self.lower_, self.upper_)
        if self.log_monetary:
            money_idx = [self.columns_.index(c) for c in MONETARY_COLS]
            Z[:, money_idx] = signed_log1p(Z[:, money_idx])
        return Z

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if list(X[ALL_COLS].columns) != self.columns_:
            raise ValueError("columns do not match those seen during fit")
        Z = (self._numeric_block(X) - self.mean_) / self.std_
        if not self.one_hot:
            return Z
        blocks = [Z]
        for col in ORDINAL_COLS:
            raw = X[col].to_numpy()[:, None]
            blocks.append((raw == self.levels_[col][None, :]).astype(float))
        return np.hstack(blocks)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def feature_names_(self) -> list[str]:
        names = list(self.columns_)
        if self.one_hot:
            for col in ORDINAL_COLS:
                names += [f"{col}={int(v)}" for v in self.levels_[col]]
        return names
