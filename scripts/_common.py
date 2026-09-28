"""Shared setup for the two training scripts, so both models are guaranteed to
see the same split, the same preprocessing and the same architecture. The only
thing that differs between them is `w_default`."""

from pathlib import Path

import numpy as np

from ml_assignment.data import Preprocessor, load_raw, train_val_split
from ml_assignment.losses import brier_score, economic_loss
from ml_assignment.mlp import predict
from ml_assignment.train import save_model, train_mlp

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

# Selected by 5-fold cross-validated random search over 30 of 150 grid points
# (see ml_assignment.tuning.SEARCH_SPACE and artifacts/tuning_*.csv).
#
# One configuration is shared by both models so that the only difference between
# them is w_default, as the assignment checklist requires ("same architecture and
# preprocessing"). It sits within one cv_sd of the best config for each objective
# separately: Brier 0.134660 vs 0.134364 best, economic 0.269672 vs 0.269598 best.
SPLIT_SEED = 0
VAL_FRAC = 0.2
PREPROCESS = dict(one_hot=True)
HYPERPARAMS = dict(hidden=(64,), lr=1e-3, l2=1e-3, batch_size=256,
                   max_epochs=300, patience=20, seed=0)


def run(name: str, w_default: float) -> dict:
    X_trn, y_trn, _ = load_raw()
    X_tr, X_val, y_tr, y_val = train_val_split(
        X_trn, y_trn, val_frac=VAL_FRAC, seed=SPLIT_SEED
    )
    pp = Preprocessor(**PREPROCESS).fit(X_tr)
    Z_tr, Z_val = pp.transform(X_tr), pp.transform(X_val)

    print(f"[{name}] training on {Z_tr.shape[0]} rows, {Z_tr.shape[1]} features, "
          f"w_default={w_default}")
    res = train_mlp(Z_tr, y_tr, Z_val, y_val, w_default=w_default, **HYPERPARAMS)

    p_val = predict(res["params"], Z_val)
    bs, econ = brier_score(y_val, p_val), economic_loss(y_val, p_val)
    print(f"[{name}] stopped at epoch {res['best_epoch']} of {res['epochs_run']} run")
    print(f"[{name}] validation Brier = {bs:.5f} | economic = {econ:.5f}")
    print(f"[{name}] mean predicted probability = {p_val.mean():.4f} "
          f"(validation default rate = {y_val.mean():.4f})")

    save_model(ARTIFACTS / f"{name}_model.npz", res["params"], res["config"])
    np.savez(ARTIFACTS / f"{name}_history.npz",
             train=np.array(res["history"]["train"]),
             val=np.array(res["history"]["val"]))
    print(f"[{name}] saved to {ARTIFACTS / f'{name}_model.npz'}")
    return res
