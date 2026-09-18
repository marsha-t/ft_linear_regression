
import numpy as np


def standardise(values):
    """
    Standardise values using z-score normalisation (mean 0 and std 1)

    Args:
        values (np.ndarray): sequence of values to standardise

    Returns:
        tuple:
            scaled (np.ndarray): standardised values
            mean (float): mean of original values
            std (float): standard deviation of original values

    Raises:
        ValueError:
            If standard deviation is zero
    """
    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        raise ValueError(
            "Cannot standardise values with zero standard deviation"
        )
    scaled = (values - mean) / std
    return scaled, mean, std
