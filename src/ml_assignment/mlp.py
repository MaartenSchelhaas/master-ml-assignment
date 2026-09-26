"""Hand-rolled multilayer perceptron: forward pass, loss, gradients, updates.

Only numpy / scipy.special are used here. No autograd, no ready-made NN
implementations, per the assignment's implementation requirements.
"""

import numpy as np

from ml_assignment.activations import Activation
from ml_assignment.losses import Loss

def init_params(layer_sizes: list[int], seed: int | None = None) -> list[dict[str, np.ndarray]]:
    """Initialize weights and biases for a fully connected network using
    Xavier (Glorot) uniform initialization.

    Args:
        layer_sizes (list[int]): The sizes of the layers, inclduing the input
         and output layer.
        seed (int | None, optional): Random seed for reproducible weight
            initialization. Defaults to None.

    Returns:
        list[dict[str, np.ndarray]]: Per-layer {"W", "b"}, one entry per
            consecutive pair in layer_sizes.
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


def copy_params(params: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    """Deep copy of a params list, used by early stopping to snapshot the
    best-so-far parameters without aliasing the arrays the optimizer keeps
    updating in place."""
    copied = []
    n_layers = len(params)
    for i in range(n_layers):
        copied.append({"W": params[i]["W"].copy(), "b": params[i]["b"].copy()})
    return copied


def forward(
    params: list[dict[str, np.ndarray]],
    X: np.ndarray,
    hidden_activation: Activation,
    output_activation: Activation,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    """Run the forward pass: hidden_activation on every hidden layer,
    output_activation on the output unit.

    Args:
        params (list[dict[str, np.ndarray]]): Per-layer weights and biases.
        X (np.ndarray): Input features, shape (n, n_features).
        hidden_activation (Activation): Applied to every hidden layer.
        output_activation (Activation): Applied to the output unit.

    Returns:
        tuple[np.ndarray, list[dict[str, np.ndarray]]]: (p_hat, cache).
            Returns predicted values, and stored values for backprop, called 
            cache. p_hat has shape (n,). cache is a list with one dict per layer
            holding whatever backward needs to recompute that layer's
            gradients: the layer's input (a_in) and pre-activation (z).
    """
    n_layers = len(params)
    cache = []
    a = X

    for i in range(n_layers):
        W = params[i]["W"]
        b = params[i]["b"]
        z = a @ W + b

        if i < n_layers - 1:
            a_next = hidden_activation.value(z)

        else:
            a_next = output_activation.value(z)

        cache.append({"a_in": a, "z": z})
        a = a_next

    #Return predictions as vector
    p_hat = a[:,0]
    return p_hat, cache



def backward(
    params: list[dict[str, np.ndarray]],
    cache: list[dict[str, np.ndarray]],
    y: np.ndarray,
    p_hat: np.ndarray,
    loss: Loss,
    hidden_activation: Activation,
    output_activation: Activation,
) -> list[dict[str, np.ndarray]]:
    """Backpropagate to get per-layer gradients.

    loss.grad(y, p_hat) returns dL/dp_hat, the first gradient at the output
    layer. 
    Args:
        params (list[dict[str, np.ndarray]]): Per-layer {"W", "b"}.
        cache (list[dict[str, np.ndarray]]): Per-layer {"a_in", "z"}, as
            returned by forward.
        y (np.ndarray): True labels (0/1), shape (n,).
        p_hat (np.ndarray): Predicted probabilities, shape (n,), as
            returned by forward.
        loss (Loss): Provides dL/dp_hat via loss.grad(y, p_hat).
        hidden_activation (Activation): Must match what forward used.
        output_activation (Activation): Must match what forward used.

    Returns:
        list[dict[str, np.ndarray]]: Per-layer {"dW", "db"}, same shape and
            order as params.
    """
    n_layers = len(params)

    #Create empty gradient list for the parameters.
    grads: list[dict[str, np.ndarray]] = []
    for i in range(n_layers):
        grads.append({})

    for i in range(n_layers - 1, -1, -1):
        if i == n_layers - 1:
            #Last layer, calculate loss
            dL_dp_hat = loss.grad(y, p_hat)
            delta = dL_dp_hat.reshape(-1, 1) * output_activation.grad(cache[i]["z"])
            # a_in = n x units_{i-1}, delta = n x units_{i} ->
            #Transpose a_in to get units_{i-1} x units_{i} matrix.
            dW = cache[i]["a_in"].T @ delta
            db = delta.sum(axis=0)
            grads[i] = {"dW": dW, "db": db}

        else:
            #Recursive gradient calculation after the first gradient
            W_next = params[i + 1]["W"]
            delta = (delta @ W_next.T) * hidden_activation.grad(cache[i]["z"])
            dW = cache[i]["a_in"].T @ delta
            db = delta.sum(axis=0)
            grads[i] = {"dW": dW, "db": db}

    return grads