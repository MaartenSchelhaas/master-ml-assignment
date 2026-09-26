"""Grid search over hyperparameters using k-fold cross-validation. Run
separately for each of the two final losses, prints every combination's
average val loss across the folds and the best one at the end.
"""

import itertools

from ml_assignment.data import fit_standardizer, k_fold_split, load_raw, standardize
from ml_assignment.losses import BrierLoss, EconomicLoss, Loss
from ml_assignment.model import MLP
from ml_assignment.optim import SGD, Adam, Optimizer

hidden_sizes_options = [[8], [16, 8], [32, 16]]
# lr ranges differ a lot per optimizer (Adam needs much smaller lr than SGD),
# so lr is tuned per optimizer instead of over one shared range.
lr_options = {
    "sgd": [0.1, 0.01],
    "adam": [0.01, 0.001],
}
batch_size_options = [32, 128]
n_epochs = 500
patience = 20
k = 5


def make_optimizer(name: str, lr: float) -> Optimizer:
    """Build the Optimizer named by name."""
    if name == "sgd":
        return SGD(lr=lr)
    elif name == "adam":
        return Adam(lr=lr)
    else:
        raise ValueError(f"unknown optimizer: {name}")


def make_grid(
    hidden_sizes_options: list[list[int]],
    lr_options: dict[str, list[float]],
    batch_size_options: list[int],
) -> list[dict]:
    """Build every combination of the given hyperparameter options."""
    # optimizer and lr are coupled (each optimizer has its own lr range), so
    # build that pairing explicitly first, then combine it with the
    # independent axes below.
    optimizer_lr_pairs = []
    for optimizer, lrs in lr_options.items():
        for lr in lrs:
            optimizer_lr_pairs.append((optimizer, lr))

    grid = []
    for hidden_sizes, (optimizer, lr), batch_size in itertools.product(
        hidden_sizes_options, optimizer_lr_pairs, batch_size_options
    ):
        grid.append({
            "hidden_sizes": hidden_sizes,
            "lr": lr,
            "batch_size": batch_size,
            "optimizer": optimizer,
        })
    return grid


def evaluate_config(
    config: dict, loss: Loss, folds: list, n_epochs: int, patience: int
) -> tuple[float, list[int]]:
    """Fit config on each of the folds, return the average val loss.

    Args:
        config (dict): One hyperparameter combination, as produced by
            make_grid.
        loss (Loss): Loss to train and evaluate with.
        folds (list): k entries of (X_tr, X_val, y_tr, y_val), as returned
            by data.k_fold_split.
        n_epochs (int): Max epochs, same for every config.
        patience (int): Early stopping patience, same for every config.

    Returns:
        tuple[float, list[int]]: (average val loss over the folds, the
            epoch each fold actually stopped at, i.e. len(history["val_loss"])
            for that fold, so a run stuck at 1-2 epochs every time is
            visibly different from one that used most of its patience).
    """
    fold_val_losses = []
    stop_epochs = []
    for X_tr, X_val, y_tr, y_val in folds:
        mean, std = fit_standardizer(X_tr)
        X_tr = standardize(X_tr, mean, std)
        X_val = standardize(X_val, mean, std)

        model = MLP(
            input_dim=X_tr.shape[1],
            hidden_sizes=config["hidden_sizes"],
            optimizer=make_optimizer(config["optimizer"], config["lr"]),
            batch_size=config["batch_size"],
            n_epochs=n_epochs,
            loss=loss,
            patience=patience,
            seed=0,
        )
        history = model.fit(X_tr, y_tr, X_val, y_val)
        fold_val_losses.append(min(history["val_loss"]))
        stop_epochs.append(len(history["val_loss"]))

    return sum(fold_val_losses) / len(fold_val_losses), stop_epochs


def tune(loss: Loss, folds: list, grid: list[dict], n_epochs: int, patience: int) -> None:
    """Evaluate every config in grid, print each one's average val loss and
    per-fold stop epoch, then print the best one.
    """
    best_val_loss = float("inf")
    best_config = None

    for config in grid:
        avg_val_loss, stop_epochs = evaluate_config(config, loss, folds, n_epochs, patience)
        print(f"{config} -> avg_val_loss={avg_val_loss:.5f} stopped_at_epochs={stop_epochs}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_config = config

    print(f"best: {best_config} -> avg_val_loss={best_val_loss:.5f}")


if __name__ == "__main__":
    X_trn, y_trn, X_test = load_raw()
    grid = make_grid(hidden_sizes_options, lr_options, batch_size_options)

    print("Brier loss:")
    tune(BrierLoss(), k_fold_split(X_trn, y_trn, k=k), grid, n_epochs, patience)

    print("\nEconomic loss:")
    tune(EconomicLoss(), k_fold_split(X_trn, y_trn, k=k), grid, n_epochs, patience)
