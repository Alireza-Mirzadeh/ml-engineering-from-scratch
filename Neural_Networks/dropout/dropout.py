# Inverted dropout implementation from scratch using numpy.

import numpy as np


def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply inverted dropout to the input array x.

    Args:
        x (list): Input array to apply dropout to.
        p (float): Dropout probability (0 <= p < 1). Default is 0.5.
        rng (np.random.Generator, optional): Random number generator for reproducibility. If None, a new random generator will be used.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the output array after applying dropout and the dropout pattern (mask) used for the operation.
    """

    # Validate the dropout probability
    if not (0 <= p < 1):
        raise ValueError("Dropout probability p must be in the range [0, 1).")

    # Convert the input list to a numpy array
    x = np.asarray(x)

    # Generate random values for dropout mask
    if rng is None:
        random_values = np.random.random(x.shape)
    else:
        random_values = rng.random(x.shape)

    # 1 = keep, 0 = drop
    mask = (random_values >= p).astype(float)

    # Inverted dropout: Scale the mask to maintain the expected value of the activations
    dropout_pattern = mask / (1 - p)

    # Apply the dropout mask to the input
    output = x * dropout_pattern

    return (output, dropout_pattern)


# Test the dropout function
if __name__ == "__main__":

    x = [[1, 2], [3, 4]]
    p = 0.5
    seed = 123
    rng = np.random.default_rng(seed)

    # Apply dropout
    output, dropout_pattern = dropout(x, p, rng)

    print("Input:\n", x)
    print("Dropout Pattern:\n", dropout_pattern)
    print("Output after Dropout:\n", output)
