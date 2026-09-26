"""Loss functions used to train and evaluate the two final models.

Loss is the base class: any loss just needs a value (for logging/scoring)
and a gradient (dL/dp_hat, the seed gradient mlp.backward starts from).
"""

from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    @abstractmethod
    def value(self, y: np.ndarray, p_hat: np.ndarray) -> float:
        """Scalar loss value, used for logging/evaluation."""
        raise NotImplementedError

    @abstractmethod
    def grad(self, y: np.ndarray, p_hat: np.ndarray) -> np.ndarray:
        """Gradient function, used in the backpropagation"""
        raise NotImplementedError


class BrierLoss(Loss):
    def value(self, y: np.ndarray, p_hat: np.ndarray) -> float:
        return float(np.mean((y - p_hat) ** 2))

    def grad(self, y: np.ndarray, p_hat: np.ndarray) -> np.ndarray:
        n = len(y)
        return -2 * (y - p_hat) / n


class EconomicLoss(Loss):
    def __init__(self, w_default: float = 3.0):
        self.w_default = w_default

    def _weights(self, y: np.ndarray) -> np.ndarray:
        return np.where(y == 1, self.w_default, 1.0)

    def value(self, y: np.ndarray, p_hat: np.ndarray) -> float:
        weights = self._weights(y)
        return float(np.mean(weights * (y - p_hat) ** 2))

    def grad(self, y: np.ndarray, p_hat: np.ndarray) -> np.ndarray:
        weights = self._weights(y)
        n = len(y)
        return -2 * weights * (y - p_hat) / n