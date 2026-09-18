import pytest
import numpy as np

from linear_regression import standardise

def test_standardise_values():
    values = np.array([1, 2, 3])

    scaled, mean, std = standardise(values)

    expected = np.array([
        (1 - 2) / std,
        (2 - 2) / std,
        (3 - 2) / std,
    ])

    assert np.allclose(scaled, expected)
    assert mean == 2
    assert np.isclose(std, np.std(values))

def test_standardised_mean_is_zero():
    values = np.array([10, 20, 30, 40])

    scaled, _, _ = standardise(values)

    assert np.isclose(np.mean(scaled), 0)

def test_standardised_std_is_one():
    values = np.array([10, 20, 30, 40])

    scaled, _, _ = standardise(values)

    assert np.isclose(np.std(scaled), 1)

def test_standardise_zero_std():
    values = np.array([5, 5, 5])

    with pytest.raises(ValueError):
        standardise(values)
        