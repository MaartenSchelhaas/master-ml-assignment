"""Public API: the MLP class. This is what scripts/*.py and hyperparameter
tuning code call, everything else in this package is an implementation
detail behind it. Plain constructor kwargs, no config object, src stays
agnostic to however external code chooses to bundle/sweep hyperparameters."""

import numpy as np

from ml_assignment import mlp
from ml_assignment.activations import ReLU, Sigmoid
from ml_assignment.losses import Loss
from ml_assignment.optim import Optimizer


class MLP:
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        optimizer: Optimizer,
        batch_size: int,
        n_epochs: int,
        loss: Loss,
        patience: int | None = None,
        seed: int | None = None,
    ):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.loss = loss
        self.patience = patience
        self.seed = seed
        self.hidden_activation = ReLU()
        self.output_activation = Sigmoid()

        layer_sizes = [input_dim, *hidden_sizes, 1]
        self.params: list[dict[str, np.ndarray]] = mlp.init_params(layer_sizes, seed=seed)

    def fit(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Fit the model, with the amount of epoch, and batch size for gradient calculation.
            Updates after each batch, new epoch once the training data is exhausted
            (data is reshuffled at the start of every epoch). If
            batch_size = n, batch gradient descent, if batch_size = 1, sgd, anything
            in between: mini-batch.
        Args:
            X_tr (np.ndarray): Training features, shape (n_train, input_dim).
            y_tr (np.ndarray): Training labels (0/1), shape (n_train,).
            X_val (np.ndarray | None, optional): Validation features, shape
                (n_val, input_dim). When given together with y_val, validation
                loss is recorded every epoch. Defaults to None.
            y_val (np.ndarray | None, optional): Validation labels, shape
                (n_val,). Defaults to None.

        Returns:
            dict[str, list[float]]: {"train_loss": per-epoch training loss,
                "val_loss": per-epoch validation loss, empty if X_val/y_val
                weren't given}.

        Note:
            If self.patience is set, X_val/y_val are required. Training then
            stops early once val_loss hasn't improved for self.patience
            epochs in a row, and self.params is rolled back to whichever
            epoch had the best val_loss.
        """
        assert X_tr.shape[1] == self.input_dim, (
            f"X_tr has {X_tr.shape[1]} features, model was built for {self.input_dim}"
        )
        if self.patience is not None:
            assert X_val is not None and y_val is not None, (
                "patience requires X_val/y_val to know when validation loss stops improving"
            )

        n = X_tr.shape[0]
        rng = np.random.default_rng(self.seed)
        #For plotting purposes
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_params = None
        epochs_without_improvement = 0

        for _ in range(self.n_epochs):
            perm = rng.permutation(n)
            X_shuffled = X_tr[perm]
            y_shuffled = y_tr[perm]

            start = 0
            while start < n:
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                p_hat_batch, cache = mlp.forward(
                    self.params, X_batch, self.hidden_activation, self.output_activation
                )
                grads = mlp.backward(
                    self.params,
                    cache,
                    y_batch,
                    p_hat_batch,
                    self.loss,
                    self.hidden_activation,
                    self.output_activation,
                )
                self.optimizer.step(self.params, grads)

                start = end

            history["train_loss"].append(self.loss.value(y_tr, self.predict_proba(X_tr)))

            if X_val is not None and y_val is not None:
                val_loss = self.loss.value(y_val, self.predict_proba(X_val))
                history["val_loss"].append(val_loss)

                if self.patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = mlp.copy_params(self.params)
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                    if epochs_without_improvement >= self.patience:
                        break

        if best_params is not None:
            self.params = best_params

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_hat, _ = mlp.forward(self.params, X, self.hidden_activation, self.output_activation)
        return p_hat
