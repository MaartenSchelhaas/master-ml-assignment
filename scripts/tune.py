"""Grid search over hyperparameters using k-fold cross-validation. Run
separately for each of the two final losses, prints every combination's
average val loss across the folds and the best one at the end.
"""

import itertools

from ml_assignment.data import fit_standardizer, k_fold_split, load_raw, standardize
from ml_assignment.losses import BrierLoss, EconomicLoss, Loss
from ml_assignment.model import MLP
from ml_assignment.optim import SGD

hidden_sizes_options = [[8], [16, 8], [32, 16]]
lr_options = [0.1, 0.01, 0.001]
batch_size_options = [32, 64, 256]
patience = 30
k = 5


def make_grid(
    hidden_sizes_options: list[list[int]], lr_options: list[float], batch_size_options: list[int]
) -> list[dict]:
    """Build every combination of the given hyperparameter options.

    Args:
        hidden_sizes_options (list[list[int]]): Candidate hidden_sizes.
        lr_options (list[float]): Candidate learning rates.
        batch_size_options (list[int]): Candidate batch sizes.

    Returns:
        list[dict]: One dict per combination, e.g.
            {"hidden_sizes": [16, 8], "lr": 0.01, "batch_size": 32}. Adding
            another hyperparameter later means adding another options list
            and another entry in itertools.product/the dict below, not
            another level of nesting.
    """
    grid = []
    for hidden_sizes, lr, batch_size in itertools.product(
        hidden_sizes_options, lr_options, batch_size_options
    ):
        grid.append({"hidden_sizes": hidden_sizes, "lr": lr, "batch_size": batch_size})
    return grid


def evaluate_config(config: dict, loss: Loss, folds: list, patience: int) -> float:
    """Fit config on each of the folds, return the average val loss.

    Args:
        config (dict): One hyperparameter combination, as produced by
            make_grid.
        loss (Loss): Loss to train and evaluate with.
        folds (list): k entries of (X_tr, X_val, y_tr, y_val), as returned
            by data.k_fold_split.
        patience (int): Early stopping patience, same for every config.

    Returns:
        float: Average, over the folds, of that fold's best val loss.
    """
    fold_val_losses = []
    for X_tr, X_val, y_tr, y_val in folds:
        mean, std = fit_standardizer(X_tr)
        X_tr = standardize(X_tr, mean, std)
        X_val = standardize(X_val, mean, std)

        model = MLP(
            input_dim=X_tr.shape[1],
            hidden_sizes=config["hidden_sizes"],
            optimizer=SGD(lr=config["lr"]),
            batch_size=config["batch_size"],
            n_epochs=300,
            loss=loss,
            patience=patience,
            seed=0,
        )
        history = model.fit(X_tr, y_tr, X_val, y_val)
        fold_val_losses.append(min(history["val_loss"]))

    return sum(fold_val_losses) / len(fold_val_losses)


def tune(loss: Loss, folds: list, grid: list[dict], patience: int) -> None:
    """Evaluate every config in grid, print each one's average val loss,
    then print the best one.
    """
    best_val_loss = float("inf")
    best_config = None

    for config in grid:
        avg_val_loss = evaluate_config(config, loss, folds, patience)
        print(f"{config} -> avg_val_loss={avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_config = config

    print(f"best: {best_config} -> avg_val_loss={best_val_loss:.5f}")


if __name__ == "__main__":
    X_trn, y_trn, X_test = load_raw()
    grid = make_grid(hidden_sizes_options, lr_options, batch_size_options)

    print("Brier loss:")
    tune(BrierLoss(), k_fold_split(X_trn, y_trn, k=k), grid, patience)

    print("\nEconomic loss:")
    tune(EconomicLoss(), k_fold_split(X_trn, y_trn, k=k), grid, patience)
