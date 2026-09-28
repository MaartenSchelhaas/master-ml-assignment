"""Produce predictions.npy: shape (n_test, 2), column 0 = Brier model,
column 1 = economic-loss model, same row order as X_test.csv.

Both final models are refit on all 25,000 labelled rows with the configuration
selected in step 5, for the epoch count that cross-validation found best. The
preprocessor is fitted on the full training set and applied unchanged to the
test set. X_test.csv contributes nothing to any fitted quantity.
"""

import sys
from pathlib import Path

import numpy as np

from ml_assignment.data import Preprocessor, load_raw
from ml_assignment.mlp import predict
from ml_assignment.train import save_model, train_full

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import HYPERPARAMS, PREPROCESS

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUT = ROOT / "predictions.npy"

# Median best epoch across the 5 CV folds (scripts/evaluate.py), +1 because
# train_mlp reports a zero-indexed epoch number.
EPOCHS = {"brier": 34 + 1, "econ": 24 + 1}
OBJECTIVES = {"brier": 1.0, "econ": 3.0}

# One network per objective, as the assignment requires. Raising this would
# average several same-architecture nets; the rules only forbid ensembling a
# network with other model classes, but check with the TA before changing it.
N_SEEDS = 1

if __name__ == "__main__":
    X_trn, y_trn, X_test = load_raw()
    y = y_trn.to_numpy().astype(float)

    pp = Preprocessor(**PREPROCESS).fit(X_trn)          # fitted on training data only
    Z_trn, Z_test = pp.transform(X_trn), pp.transform(X_test)
    print(f"train {Z_trn.shape}, test {Z_test.shape}, config {HYPERPARAMS}")

    columns = []
    for name, w_default in OBJECTIVES.items():
        runs = []
        for s in range(N_SEEDS):
            params = train_full(
                Z_trn, y, epochs=EPOCHS[name], w_default=w_default,
                hidden=HYPERPARAMS["hidden"], lr=HYPERPARAMS["lr"],
                l2=HYPERPARAMS["l2"], batch_size=HYPERPARAMS["batch_size"],
                seed=HYPERPARAMS["seed"] + s,
            )
            runs.append(predict(params, Z_test))
            if s == 0:
                save_model(ARTIFACTS / f"final_{name}_model.npz", params,
                           {**HYPERPARAMS, "w_default": w_default,
                            "epochs": EPOCHS[name], "n_features": Z_trn.shape[1]})
        p_test = np.mean(runs, axis=0)
        columns.append(p_test)
        p_trn = predict(params, Z_trn)
        print(f"[{name}] {EPOCHS[name]} epochs | test mean {p_test.mean():.4f} "
              f"(in-sample train mean {p_trn.mean():.4f}) | "
              f"test range [{p_test.min():.4f}, {p_test.max():.4f}]")

    predictions = np.column_stack(columns)

    assert predictions.shape == (len(X_test), 2), predictions.shape
    assert np.isfinite(predictions).all(), "non-finite entries"
    assert (predictions >= 0).all() and (predictions <= 1).all(), "outside [0, 1]"

    np.save(OUT, predictions)
    print(f"\nsaved {OUT}")
    print(f"  shape {predictions.shape}, dtype {predictions.dtype}")
    print(f"  col 0 (Brier)    mean {predictions[:,0].mean():.4f}  "
          f"min {predictions[:,0].min():.4f}  max {predictions[:,0].max():.4f}")
    print(f"  col 1 (Economic) mean {predictions[:,1].mean():.4f}  "
          f"min {predictions[:,1].min():.4f}  max {predictions[:,1].max():.4f}")
    print(f"  all finite: {np.isfinite(predictions).all()} | "
          f"within [0,1]: {bool(((predictions >= 0) & (predictions <= 1)).all())}")
