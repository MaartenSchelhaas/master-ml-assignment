"""Public API: the MLP class and its hyperparameters. This is what
scripts/*.py and hyperparameter tuning code call, everything else in this
package is an implementation detail behind it."""

from dataclasses import dataclass

import numpy as np

from ml_assignment import losses, mlp, optim


@dataclass
class MLPConfig:
    hidden_sizes: list[int]
    lr: float
    n_epochs: int
    loss: str  # "brier" or "econ"
    w_default: float = 3.0
    seed: int | None = None


class MLP:
    def __init__(self, config: MLPConfig):
        self.config = config
        self.params: list[dict] | None = None

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
        losses.brier_score or losses.economic_loss to match config.loss.
        Returns {"train_loss": [...], "val_loss": [...]}."""
        layer_sizes = [X_tr.shape[1], *self.config.hidden_sizes, 1]
        self.params = mlp.init_params(layer_sizes, seed=self.config.seed)

        history: dict = {"train_loss": [], "val_loss": []}
        weights = losses.sample_weights(y_tr, self.config.loss, self.config.w_default)
        score_fn = losses.brier_score if self.config.loss == "brier" else losses.economic_loss

        for _ in range(self.config.n_epochs):
            p_hat, cache = mlp.forward(self.params, X_tr)
            grads = mlp.backward(self.params, cache, y_tr, weights)
            optim.sgd_step(self.params, grads, self.config.lr)

            history["train_loss"].append(score_fn(y_tr, p_hat))
            if X_val is not None and y_val is not None:
                history["val_loss"].append(score_fn(y_val, self.predict_proba(X_val)))

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_hat, _ = mlp.forward(self.params, X)
        return p_hat