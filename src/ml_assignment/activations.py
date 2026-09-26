"""Activation functions, paired with their own derivative so forward and
backward always use a matched (value, grad) pair instead of relying on
separate functions kept in sync by naming convention."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.special import expit


class Activation(ABC):
    @abstractmethod
    def value(self, z: np.ndarray) -> np.ndarray:
        """The activation itself, applied elementwise to the pre-activation z."""
        raise NotImplementedError

    @abstractmethod
    def grad(self, z: np.ndarray) -> np.ndarray:
        """Elementwise derivative of the activation, evaluated at the same z."""
        raise NotImplementedError


class ReLU(Activation):
    def value(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(z, 0)

    def grad(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)


class Sigmoid(Activation):
    def value(self, z: np.ndarray) -> np.ndarray:
        return expit(z)

    def grad(self, z: np.ndarray) -> np.ndarray:
        s = self.value(z)
        return s * (1 - s)