# Implement one update step of RMSProp optimizer.

import numpy as np


def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Perform one update step of the RMSProp optimizer.

    Args:
        w (list): Current weights.
        g (list): Current gradients.
        s (list): Current moving average of squared gradients.
        lr (float): Learning rate.
        beta (float): Decay rate for the moving average of squared gradients.
        eps (float): Small constant to prevent division by zero.

    Returns:
        tuple[list, list]: Updated weights and updated moving average of squared gradients.
    """
    # Convert inputs to numpy arrays for easier manipulation
    w = np.array(w)
    g = np.array(g)
    s = np.array(s)

    # Update the moving average of squared gradients
    new_s = beta * s + (1 - beta) * g**2

    # Update the weights
    new_w = w - lr * g / (np.sqrt(new_s) + eps)

    # Return tuple of updated weights and updated moving average of squared gradients as lists
    return (new_w.tolist(), new_s.tolist())


# Test the function 
if __name__ == "__main__":
    w = ([1.0, 2.0],)
    g = ([0.2, -0.4],)
    s = ([0.0, 0.0],)
    lr = 0.1
    beta = 0.9
    eps = 1e-8
    new_w, new_s = rmsprop_step(w, g, s, lr, beta, eps)

    print("New weights:", new_w)
    print("New moving average of squared gradients:", new_s)
