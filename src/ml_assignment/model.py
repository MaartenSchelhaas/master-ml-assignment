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
        n_epochs: int,
        loss: Loss,
        seed: int | None = None,
    ):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.lr = lr
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
        """Full-batch gradient descent: one forward/backward/update per
        epoch over all of X_tr (no minibatching for now, that's a later
        addition if speed or memory become an issue). Records train loss,
        and val loss if X_val/y_val are given, after every epoch using
        self.loss.value. Returns {"train_loss": [...], "val_loss": [...]}."""
        assert X_tr.shape[1] == self.input_dim, (
            f"X_tr has {X_tr.shape[1]} features, model was built for {self.input_dim}"
        )

        history: dict = {"train_loss": [], "val_loss": []}

        for _ in range(self.n_epochs):
            p_hat, cache = mlp.forward(
                self.params, X_tr, self.hidden_activation, self.output_activation
            )
            grads = mlp.backward(
                self.params, cache, y_tr, p_hat, self.loss, self.hidden_activation, self.output_activation
            )
            optim.sgd_step(self.params, grads, self.lr)

            history["train_loss"].append(self.loss.value(y_tr, p_hat))
            if X_val is not None and y_val is not None:
                history["val_loss"].append(self.loss.value(y_val, self.predict_proba(X_val)))

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_hat, _ = mlp.forward(self.params, X, self.hidden_activation, self.output_activation)
        return p_hat
