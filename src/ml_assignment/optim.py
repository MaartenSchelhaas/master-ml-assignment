"""Parameter update rules, kept separate from mlp.py so the forward/backward
math has no notion of a learning rate, and so other update rules can be
added later without touching model.py's training loop."""


def sgd_step(params: list[dict], grads: list[dict], lr: float) -> None:
    """In-place plain SGD update: W -= lr * dW, b -= lr * db, per layer."""
    raise NotImplementedError