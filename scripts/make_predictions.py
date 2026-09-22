"""Produce predictions.npy: shape (n_test, 2), column 0 = Brier model,
column 1 = economic-loss model, same row order as X_test.csv."""

import numpy as np

from ml_assignment.data import load_raw

if __name__ == "__main__":
    X_trn, y_trn, X_test = load_raw()
    # p_brier = ...
    # p_econ = ...
    # predictions = np.column_stack([p_brier, p_econ])
    # np.save("predictions.npy", predictions)
    raise NotImplementedError
