"""Public API: the MLP class. This is what scripts/*.py and hyperparameter
tuning code call, everything else in this package is an implementation
detail behind it. Plain constructor kwargs, no config object, src stays
agnostic to however external code chooses to bundle/sweep hyperparameters."""

import numpy as np

from ml_assignment import mlp, optim
from ml_assignment.activations import ReLU, Sigmoid
from ml_assignment.losses import Loss


class MLP:
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        lr: float,
        batch_size: int,
        n_epochs: int,
        loss: Loss,
        seed: int | None = None,
    ):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.loss = loss
        self.seed = seed
        self.hidden_activation = ReLU()
        self.output_activation = Sigmoid()

        layer_sizes = [input_dim, *hidden_sizes, 1]
        self.params: list[dict] = mlp.init_params(layer_sizes, seed=seed)

    def fit(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict:
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
            dict: {"train_loss": per-epoch training loss, "val_loss": per-epoch
                validation loss, empty if X_val/y_val weren't given}.
        """
        assert X_tr.shape[1] == self.input_dim, (
            f"X_tr has {X_tr.shape[1]} features, model was built for {self.input_dim}"
        )

        n = X_tr.shape[0]
        rng = np.random.default_rng(self.seed)
        history: dict = {"train_loss": [], "val_loss": []}

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
                optim.sgd_step(self.params, grads, self.lr)

                start = end

            history["train_loss"].append(self.loss.value(y_tr, self.predict_proba(X_tr)))
            if X_val is not None and y_val is not None:
                history["val_loss"].append(self.loss.value(y_val, self.predict_proba(X_val)))

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_hat, _ = mlp.forward(self.params, X, self.hidden_activation, self.output_activation)
        return p_hat
