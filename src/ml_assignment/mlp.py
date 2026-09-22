"""Hand-rolled multilayer perceptron: forward pass, loss, gradients, updates.

Only numpy / scipy.special are used here. No autograd, no ready-made NN
implementations, per the assignment's implementation requirements.
"""

import numpy as np
from scipy.special import expit

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
    """Run the forward pass.
    Returns predicted probabilities and cached activations needed for
    backprop.
    """
    cache = []
    A = X
    num_layers = len(params)

    for l, layer in enumerate(params):
        W, b = layer["W"], layer["b"]
        Z = A @ W + b
        cache.append({"A_prev": A, "Z": Z})

        if l == num_layers - 1:
            # Output layer: bounded probability output via logistic sigmoid
            A = expit(Z)
        else:
            # Hidden layer: rectified linear unit (ReLU)
            A = np.maximum(0.0, Z)

    return A, cache


def backward(
    params: list[dict], cache: list[dict], y: np.ndarray, weights: np.ndarray
) -> list[dict]:
    """Backpropagate the weighted squared-error loss to get gradients."""
    n = y.shape[0]
    y = np.asarray(y).reshape(-1, 1)
    weights = np.asarray(weights).reshape(-1, 1)

    num_layers = len(params)
    grads = [{} for _ in range(num_layers)]

    # Final layer predictions: p_hat = sigma(Z_L)
    p_hat = expit(cache[-1]["Z"])

    # Output error: dL/dZ_L = (dL/dp_hat) * (dp_hat/dZ_L)
    # dL/dp_hat = (-2/n) * w * (y - p_hat)
    # dp_hat/dZ_L = p_hat * (1 - p_hat)
    dZ = (-2.0 / n) * weights * (y - p_hat) * p_hat * (1.0 - p_hat)

    for l in reversed(range(num_layers)):
        A_prev = cache[l]["A_prev"]

        # Gradients for weights and biases of layer l
        grads[l]["W"] = A_prev.T @ dZ
        grads[l]["b"] = np.sum(dZ, axis=0)

        # Propagate error to previous layer if not at the input layer
        if l > 0:
            dA_prev = dZ @ params[l]["W"].T
            Z_prev = cache[l - 1]["Z"]
            # ReLU gradient: 1 if pre-activation > 0, else 0
            dZ = dA_prev * (Z_prev > 0.0)

    return grads


def sgd_update(params: list[dict], grads: list[dict], lr: float) -> None:
    """In-place parameter update."""
    for p, g in zip(params, grads):
        p["W"] -= lr * g["W"]
        p["b"] -= lr * g["b"]
