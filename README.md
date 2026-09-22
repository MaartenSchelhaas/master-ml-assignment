# FEM21045 ML Assignment

Two hand-rolled neural networks predicting default probability: one trained on the
Brier score, one on the asymmetric economic loss (defaults weighted 3x).

## Setup

Install [uv](https://docs.astral.sh/uv/) if you don't have it, then from the repo root:

```
uv sync
```

This creates a `.venv` and installs everything in `pyproject.toml`. No need to
`pip install` anything by hand, and no need to manually create or activate a
virtualenv, `uv run` (below) handles that.

To add a new dependency later:

```
uv add <package>
```

This updates `pyproject.toml` and `uv.lock` automatically, commit both.

## Data

Download `X_trn.csv`, `y_trn.csv`, and `X_test.csv` from Canvas and put them in
`data/`. That folder is gitignored, don't commit the data.

## Running things

Prefix any Python command with `uv run` so it uses the project's environment:

```
uv run python scripts/train_brier.py
uv run python scripts/train_econ.py
uv run python scripts/make_predictions.py
uv run jupyter lab
```

## Structure

```
data/                   CSVs from Canvas (gitignored)
src/ml_assignment/
  mlp.py                forward pass, backprop, parameter updates (numpy only)
  losses.py             Brier score and economic loss
  data.py                loading and train/validation splitting
scripts/
  train_brier.py        trains the Brier-score model
  train_econ.py          trains the economic-loss model
  make_predictions.py   writes predictions.npy (n_test, 2): col 0 = Brier model,
                         col 1 = economic-loss model
report.tex              LaTeX report template (add from Canvas)
```

## Rules to keep in mind

- Forward pass, loss, gradients and parameter updates must be written by us using
  plain numpy/scipy array arithmetic, no autograd, no sklearn/torch/etc.
- Never touch `X_test.csv` for model selection, tuning, or preprocessing decisions,
  validation only.
- Every preprocessing transform must be fit on the training data only.