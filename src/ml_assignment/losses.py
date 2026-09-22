"""Loss functions used to train and evaluate the two final models."""

import numpy as np


def brier_score(y: np.ndarray, p_hat: np.ndarray) -> float:
    return float(np.mean((y - p_hat) ** 2))


def economic_loss(y: np.ndarray, p_hat: np.ndarray, w_default: float = 3.0) -> float:
    weights = np.where(y == 1, w_default, 1.0)
    return float(np.mean(weights * (y - p_hat) ** 2))


def sample_weights(y: np.ndarray, kind: str, w_default: float = 3.0) -> np.ndarray:
    """Per-sample weights for the weighted squared-error loss used in
    training: all 1s for "brier", w_default on defaults and 1 elsewhere
    for "econ". Shared by MLP.fit's backward pass and by economic_loss
    above so the weighting logic isn't duplicated."""
    raise NotImplementedError
