"""Hand-rolled multilayer perceptron: forward pass, loss, gradients, updates.

Only numpy / scipy.special are used here. No autograd, no ready-made NN
implementations, per the assignment's implementation requirements.
"""

from collections.abc import Callable

import numpy as np
from scipy.special import expit

def init_params(layer_sizes: list[int], seed: int | None = None) -> list[dict]:
    """Initialize weights and biases for a fully connected network using
    Xavier (Glorot) uniform initialization: W ~ Uniform(-limit, limit) with
    limit = sqrt(6 / (n_in + n_out)), b = 0.

    layer_sizes includes the input dimension and the output dimension,
    e.g. [n_features, 16, 8, 1].
    """
    rng = np.random.default_rng(seed)
    params = []
    n_layers = len(layer_sizes) - 1
    for i in range(n_layers):
        n_in = layer_sizes[i]
        n_out = layer_sizes[i + 1]
        limit = np.sqrt(6.0 / (n_in + n_out))
        W = rng.uniform(-limit, limit, size=(n_in, n_out))
        b = np.zeros(n_out)
        params.append({"W": W, "b": b})
    return params


def relu(z: np.ndarray) -> np.ndarray:
    """Elementwise ReLU, used for every hidden layer's activation."""
    return np.maximum(z,0)



def relu_grad(z: np.ndarray) -> np.ndarray:
    """Elementwise derivative of ReLU at the pre-activation z, 1 where
    z > 0 and 0 otherwise."""
    return (z > 0).astype(float)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Elementwise sigmoid via scipy.special.expit, used on the output
    unit so predictions are valid probabilities."""
    return expit(z)


def sigmoid_grad(z: np.ndarray) -> np.ndarray:
    """Elementwise derivative of sigmoid at the pre-activation z."""
    s = sigmoid(z)
    return s * (1 - s)


def forward(params: list[dict], X: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Run the forward pass: ReLU on every hidden layer, sigmoid on the
    output unit. Returns (p_hat, cache), where p_hat has shape (n,) and
    cache is a list with one dict per layer holding whatever backward
    needs to recompute that layer's gradients, e.g. the layer's input,
    pre-activation, and activation."""
    #for i in len(params)
    n_layers = len(params)
    cache = []
    a = X

    for i in range(n_layers):
        W = params[i]["W"]
        b = params[i]["b"]
        z = a @ W + b 

        if i < n_layers - 1:
            a_next = relu(z)

        else: 
            a_next = sigmoid(z)

        cache.append({"a_in": a, "z": z, "a_out": a_next})
        a = a_next

    #Return predictions as vector
    p_hat = a[:,0]
    return p_hat, cache



def backward(
    params: list[dict],
    cache: list[dict],
    y: np.ndarray,
    loss_grad: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> list[dict]:
    """Backpropagate to get per-layer gradients.

    loss_grad(y, p_hat) returns dL/dp_hat, the seed gradient at the output
    layer. backward doesn't need to know which loss produced it, or how
    it's weighted internally, only its derivative w.r.t. p_hat.
    """
    n_layers = len(params)

    #Create empty gradient list for the parameters. 
    grads: list[dict] = []
    for i in range(n_layers):
        grads.append({})

    for i in range(n_layers - 1, -1, -1):
        if i == n_layers - 1:
            #Last layer, calculate loss
            p_hat = cache[i]["a_out"].reshape(-1)
            dL_dp_hat = loss_grad(y, p_hat)
            delta = dL_dp_hat.reshape(-1, 1) * sigmoid_grad(cache[i]["z"])
            # a_in = n x units_{i-1}, delta = n x units_{i} -> 
            #Transpose a_in to get units_{i-1} x units_{i} matrix. 
            dW = cache[i]["a_in"].T @ delta
            db = delta.sum(axis=0)
            grads[i] = {"dW": dW, "db": db}

        else:
            W_next = params[i + 1]["W"]
            delta = (delta @ W_next.T) * relu_grad(cache[i]["z"])
            dW = cache[i]["a_in"].T @ delta
            db = delta.sum(axis=0)
            grads[i] = {"dW": dW, "db": db}

    return grads
