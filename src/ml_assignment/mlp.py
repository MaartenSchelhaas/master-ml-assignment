"""Hand-rolled multilayer perceptron: forward pass, loss, gradients, updates.

Only numpy / scipy.special are used here. No autograd, no ready-made NN
implementations, per the assignment's implementation requirements.

The network is fully connected with ReLU hidden layers and a single sigmoid
output unit, so the output is a probability in (0, 1). Both final models share
this code: the Brier model trains with all weights equal to 1, the economic
model with weight 3 on the defaults. The weights enter only through `backward`.
"""

import numpy as np
from scipy.special import expit


def init_params(
    layer_sizes: list[int], seed: int | None = None, output_bias: float | None = None
) -> list[dict]:
    """Initialize weights and biases for a fully connected network.

    layer_sizes includes the input dimension and the output dimension,
    e.g. [n_features, 16, 8, 1].

    He initialization for the ReLU hidden layers. `output_bias` sets the bias of
    the final unit, so passing logit(base rate) makes the untrained network
    predict the base rate. That matters here: the squared-error gradient carries
    a p(1 - p) factor, so a network that starts far from the base rate starts on
    a flat part of the loss surface.
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
    if output_bias is not None:
        params[-1]["b"][:] = output_bias
    return params


def forward(params: list[dict], X: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Run the forward pass. Returns predicted probabilities and cached
    activations needed for backprop.

    p has shape (n,). The cache holds, per layer, the input activation that
    layer saw and its pre-activation Z, which is what `backward` needs.
    """
    cache = []
    A = X
    n_layers = len(params)
    for i, layer in enumerate(params):
        Z = A @ layer["W"] + layer["b"]
        cache.append({"A_in": A, "Z": Z})
        A = Z if i == n_layers - 1 else np.maximum(Z, 0.0)
    p = expit(cache[-1]["Z"]).ravel()
    return p, cache


def backward(
    params: list[dict],
    cache: list[dict],
    y: np.ndarray,
    weights: np.ndarray,
    l2: float = 0.0,
) -> list[dict]:
    """Backpropagate the weighted squared-error loss to get gradients.

    The loss is L = (1/n) * sum_i w_i (y_i - p_i)^2, matching `losses.py`. With
    p = sigmoid(z) the output delta is

        dL/dz = 2 * w * (p - y) * p * (1 - p) / n,

    where the p(1 - p) factor is the sigmoid derivative. `l2` adds a penalty
    l2 * sum(W^2) on the weights only, never on the biases.
    """
    n = y.shape[0]
    p = expit(cache[-1]["Z"]).ravel()
    delta = (2.0 * weights * (p - y) * p * (1.0 - p) / n).reshape(-1, 1)

    grads = [None] * len(params)
    for i in reversed(range(len(params))):
        A_in = cache[i]["A_in"]
        dW = A_in.T @ delta
        if l2:
            dW = dW + 2.0 * l2 * params[i]["W"]
        grads[i] = {"W": dW, "b": delta.sum(axis=0)}
        if i > 0:
            delta = (delta @ params[i]["W"].T) * (cache[i - 1]["Z"] > 0)
    return grads


def sgd_update(params: list[dict], grads: list[dict], lr: float) -> None:
    """In-place parameter update."""
    for layer, grad in zip(params, grads):
        layer["W"] -= lr * grad["W"]
        layer["b"] -= lr * grad["b"]


def init_adam_state(params: list[dict]) -> dict:
    """First and second moment accumulators for `adam_update`."""
    return {
        "m": [{k: np.zeros_like(v) for k, v in layer.items()} for layer in params],
        "v": [{k: np.zeros_like(v) for k, v in layer.items()} for layer in params],
        "t": 0,
    }


def adam_update(
    params: list[dict],
    grads: list[dict],
    state: dict,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """In-place Adam update, written out rather than taken from a library.

    Preferred over plain SGD here because squared error on a sigmoid output
    produces very small gradients for confidently wrong units; the per-parameter
    rescaling keeps those units moving.
    """
    state["t"] += 1
    t = state["t"]
    bias1 = 1.0 - beta1**t
    bias2 = 1.0 - beta2**t
    for layer, grad, m, v in zip(params, grads, state["m"], state["v"]):
        for k in ("W", "b"):
            m[k] *= beta1
            m[k] += (1.0 - beta1) * grad[k]
            v[k] *= beta2
            v[k] += (1.0 - beta2) * grad[k] ** 2
            layer[k] -= lr * (m[k] / bias1) / (np.sqrt(v[k] / bias2) + eps)


def predict(params: list[dict], X: np.ndarray) -> np.ndarray:
    """Predicted probabilities, shape (n,). Convenience wrapper over `forward`."""
    return forward(params, X)[0]
