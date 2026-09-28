"""Run the cross-validated hyperparameter search for both objectives.

Each objective is tuned under its own loss and logged to its own CSV, which
doubles as the candidate-value table the report has to include.
"""

import sys
from pathlib import Path

from ml_assignment.data import load_raw
from ml_assignment.tuning import SEARCH_SPACE, run_search

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
N_CONFIGS = 30
N_SPLITS = 5

if __name__ == "__main__":
    X_trn, y_trn, _ = load_raw()
    y = y_trn.to_numpy().astype(float)

    print(f"search space: {SEARCH_SPACE}")
    print(f"sampling {N_CONFIGS} configurations, {N_SPLITS}-fold CV\n")

    for name, w_default in (("brier", 1.0), ("econ", 3.0)):
        print(f"=== tuning {name} (w_default={w_default}) ===")
        table = run_search(X_trn, y, w_default=w_default,
                           n_configs=N_CONFIGS, n_splits=N_SPLITS)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        out = ARTIFACTS / f"tuning_{name}.csv"
        table.to_csv(out, index=False)
        print(f"\n--- {name}: top 5 ---")
        print(table.head(5).to_string(index=False))
        print(f"saved {out}\n", flush=True)
