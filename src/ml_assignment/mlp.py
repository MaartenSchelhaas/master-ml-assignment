"""Hand-rolled multilayer perceptron: forward pass, loss, gradients, updates.

Only numpy / scipy.special are used here. No autograd, no ready-made NN
implementations, per the assignment's implementation requirements.
"""

import numpy as np


def init_params(layer_sizes: list[int], seed: int | None = None) -> list[dict]:
    """Initialize weights and biases for a fully connected network.

    layer_sizes includes the input dimension and the output dimension,
    e.g. [n_features, 16, 8, 1].
    """
    rng = np.random.default_rng(seed)
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        scale = np.sqrt(2.0 / n_in)
        params.append(
            {
                "W": rng.normal(0.0, scale, size=(n_in, n_out)),
                "b": np.zeros(n_out),
            }
        )
    return params


def relu(z: np.ndarray) -> np.ndarray:
    """Elementwise ReLU, used for every hidden layer's activation."""
    raise NotImplementedError


def relu_grad(z: np.ndarray) -> np.ndarray:
    """Elementwise derivative of ReLU at the pre-activation z, 1 where
    z > 0 and 0 otherwise."""
    raise NotImplementedError


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Elementwise sigmoid via scipy.special.expit, used on the output
    unit so predictions are valid probabilities."""
    raise NotImplementedError


def forward(params: list[dict], X: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Run the forward pass: ReLU on every hidden layer, sigmoid on the
    output unit. Returns (p_hat, cache), where p_hat has shape (n,) and
    cache is a list with one dict per layer holding whatever backward
    needs to recompute that layer's gradients, e.g. the layer's input,
    pre-activation, and activation."""
    raise NotImplementedError


def backward(
    params: list[dict], cache: list[dict], y: np.ndarray, weights: np.ndarray
) -> list[dict]:
    """Backpropagate the weighted squared-error loss
    L = mean(weights * (y - p_hat) ** 2) to get gradients.

    """
    raise NotImplementedError
