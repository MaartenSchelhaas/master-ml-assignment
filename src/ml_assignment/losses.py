"""Loss functions used to train and evaluate the two final models."""

import numpy as np


def brier_score(y: np.ndarray, p_hat: np.ndarray) -> float:
    return float(np.mean((y - p_hat) ** 2))


def brier_grad(y: np.ndarray, p_hat: np.ndarray) -> np.ndarray:
    """dL/dp_hat for brier_score, the seed gradient backward starts from."""
    n = len(y)
    return -2 * (y - p_hat) / n


def economic_loss(y: np.ndarray, p_hat: np.ndarray, w_default: float = 3.0) -> float:
    weights = np.where(y == 1, w_default, 1.0)
    return float(np.mean(weights * (y - p_hat) ** 2))


def economic_grad(
    y: np.ndarray, p_hat: np.ndarray, w_default: float = 3.0
) -> np.ndarray:
    """dL/dp_hat for economic_loss, the seed gradient backward starts from."""
    weights = np.where(y == 1, w_default, 1.0)
    n = len(y)
    return -2 * weights * (y - p_hat) / n
