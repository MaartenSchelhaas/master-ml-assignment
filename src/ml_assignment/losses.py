"""Loss functions used to train and evaluate the two final models.

Both objectives are the same weighted squared error,

    L = (1/n) * sum_i w_i (y_i - p_i)^2,

with w_i = 1 everywhere for the Brier score and w_i = 3 on the defaults for the
economic loss. `weights_for` is the single definition of those weights, so the
training loop and the evaluation code cannot drift apart.
"""

import numpy as np


def weights_for(y: np.ndarray, w_default: float = 1.0) -> np.ndarray:
    """Per-observation weights: `w_default` on defaults, 1 on non-defaults.

    w_default=1.0 gives the Brier score, w_default=3.0 the economic loss.
    """
    return np.where(y == 1, float(w_default), 1.0)


def weighted_mse(y: np.ndarray, p_hat: np.ndarray, weights: np.ndarray) -> float:
    """The shared objective, given an explicit weight vector."""
    return float(np.mean(weights * (y - p_hat) ** 2))


def brier_score(y: np.ndarray, p_hat: np.ndarray) -> float:
    return float(np.mean((y - p_hat) ** 2))


def economic_loss(y: np.ndarray, p_hat: np.ndarray, w_default: float = 3.0) -> float:
    return weighted_mse(y, p_hat, weights_for(y, w_default))
