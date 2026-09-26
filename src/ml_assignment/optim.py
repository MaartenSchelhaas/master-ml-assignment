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
        """Plain SGD update.

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


class Adam(Optimizer):
    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        # Per-parameter first/second moment estimates, same shape as
        # params. Can't allocate them here, we don't know the param shapes
        # yet, so do it lazily on the first step() call instead.
        self.m: list[dict[str, np.ndarray]] | None = None
        self.v: list[dict[str, np.ndarray]] | None = None

    def step(self, params: list[dict[str, np.ndarray]], grads: list[dict[str, np.ndarray]]) -> None:
        """Adam update.

        Args:
            params (list[dict[str, np.ndarray]]): per-layer {"W", "b"}, same
                shape as grads, updated in place.
            grads (list[dict[str, np.ndarray]]): per-layer {"dW", "db"} from
                mlp.backward, same shapes as the matching params entry.
        """
        n_layers = len(params)

        if self.m is None:
            self.m = []
            self.v = []
            for i in range(n_layers):
                self.m.append({"W": np.zeros_like(params[i]["W"]), "b": np.zeros_like(params[i]["b"])})
                self.v.append({"W": np.zeros_like(params[i]["W"]), "b": np.zeros_like(params[i]["b"])})

        self.t += 1
        assert self.m is not None and self.v is not None

        for i in range(n_layers):
            self.m[i]["W"] = self.beta1 * self.m[i]["W"] + (1 - self.beta1) * grads[i]["dW"]
            self.m[i]["b"] = self.beta1 * self.m[i]["b"] + (1 - self.beta1) * grads[i]["db"]

            self.v[i]["W"] = self.beta2 * self.v[i]["W"] + (1 - self.beta2) * grads[i]["dW"] ** 2
            self.v[i]["b"] = self.beta2 * self.v[i]["b"] + (1 - self.beta2) * grads[i]["db"] ** 2

            m_hat_W = self.m[i]["W"] / (1 - self.beta1**self.t)
            m_hat_b = self.m[i]["b"] / (1 - self.beta1**self.t)
            v_hat_W = self.v[i]["W"] / (1 - self.beta2**self.t)
            v_hat_b = self.v[i]["b"] / (1 - self.beta2**self.t)

            params[i]["W"] -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.eps)
            params[i]["b"] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)