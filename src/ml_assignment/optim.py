"""Parameter update rules, kept separate from mlp.py so the forward/backward
math has no notion of a learning rate. Optimizer is the base class: MLP only
needs an object with a step(params, grads) method, it doesn't need to know
which update rule that is, so other rules (momentum, etc.) can be added
later as another Optimizer subclass without touching model.py."""

from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def step(self, params: list[dict[str, np.ndarray]], grads: list[dict[str, np.ndarray]]) -> None:
        """In-place parameter update given the current gradients."""
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float):
        self.lr = lr

    def step(self, params: list[dict[str, np.ndarray]], grads: list[dict[str, np.ndarray]]) -> None:
        """Plain SGD update: W -= lr * dW, b -= lr * db, per layer.

        Args:
            params (list[dict[str, np.ndarray]]): per-layer {"W", "b"}, same
                shape as grads, updated in place.
            grads (list[dict[str, np.ndarray]]): per-layer {"dW", "db"} from
                mlp.backward, same shapes as the matching params entry.
        """
        n_layers = len(params)
        for i in range(n_layers):
            params[i]["W"] -= self.lr * grads[i]["dW"]
            params[i]["b"] -= self.lr * grads[i]["db"]