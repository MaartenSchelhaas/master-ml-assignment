# Loan Default Prediction

Two neural networks, written from scratch in numpy, predicting loan default probability: one trained on the Brier score, one on an asymmetric economic 
loss that weights missed defaults 3x over false alarms.

## Running

```
uv run python scripts/tune.py                        # k-fold grid search, prints best config per loss
uv run python scripts/make_predictions.py             # trains both final models, writes predictions.npy
uv run python scripts/make_predictions_ensembling.py  # same, as a 20-member ensemble
```

## Structure

```
data/                              training/test CSVs
src/ml_assignment/
  mlp.py                           forward pass, backprop, parameter updates
  model.py                         MLP class: fit/predict_proba API
  activations.py                   ReLU, Sigmoid
  losses.py                        Brier score, economic loss
  optim.py                         SGD, Adam
  data.py                          loading, one-hot encoding, standardizing, splitting
scripts/
  tune.py                          hyperparameter search
  make_predictions.py              final models -> predictions.npy
  make_predictions_ensembling.py   final models as an ensemble -> predictions_ensemble.npy
output/                            metrics, plots, predictions (gitignored)
docs/main.tex                      report
```

## Architecture

Fully-connected MLP with ReLU hidden layers and a sigmoid output producing
the default probability. Hidden-layer sizes, learning rate, batch size and
optimizer are chosen per loss by 5-fold cross-validation. Nominal
categorical features are one-hot encoded, everything else standardized
(fit on training data only), and training uses mini-batch gradient descent
with early stopping on a validation split.
