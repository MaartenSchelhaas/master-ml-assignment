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


def forward(params: list[dict], X: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Run the forward pass. Returns predicted probabilities and cached
    activations needed for backprop."""
    raise NotImplementedError


def backward(
    params: list[dict], cache: list[dict], y: np.ndarray, weights: np.ndarray
) -> list[dict]:
    """Backpropagate the weighted squared-error loss to get gradients."""
    raise NotImplementedError


def sgd_update(params: list[dict], grads: list[dict], lr: float) -> None:
    """In-place parameter update."""
    raise NotImplementedError
