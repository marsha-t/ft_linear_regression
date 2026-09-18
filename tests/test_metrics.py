import numpy as np
import pytest

from linear_regression.metrics import mse, mae, r2_score

# Test mse()
@pytest.mark.parametrize(
    ("y_actual, y_pred, expected"),
    [
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            0.0
        ),
        (
            np.array([1, 2, 3]),
            np.array([2, 3, 4]),
            1.0
        ),
        (
            np.array([10, 20]),
            np.array([13, 14]),
            22.5
        ),
    ],
)
def test_mse(y_actual, y_pred, expected):
    assert mse(y_actual, y_pred) == pytest.approx(expected)


# Test mae() 
@pytest.mark.parametrize(
    ("y_actual, y_pred, expected"),
    [
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            0.0
        ),
        (
            np.array([1, 2, 3]),
            np.array([2, 3, 4]),
            1.0
        ),
        (
            np.array([10, 20]),
            np.array([13, 14]),
            4.5
        ),
    ],
)
def test_mae(y_actual, y_pred, expected):
    assert mae(y_actual, y_pred) == pytest.approx(expected)


# Test r2_score() 
def test_r2_score_perfect_predictions():
    y_actual = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4])

    assert r2_score(y_actual, y_pred) == pytest.approx(1.0)


def test_r2_score_imperfect_predictions():
    y_actual = np.array([1, 2, 3, 4])
    y_pred = np.array([2, 2, 3, 5])

    expected = 0.6

    assert r2_score(y_actual, y_pred) == pytest.approx(expected)


def test_r2_score_constant_actual_values():
    y_actual = np.array([5, 5, 5])
    y_pred = np.array([5, 5, 5])

    with pytest.raises(ValueError):
        r2_score(y_actual, y_pred)