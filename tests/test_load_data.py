import pytest
import pandas as pd
import numpy as np

from linear_regression import validate_csv, load_training_data


# Test validate_csV()
def test_validate_csv_valid():
    data = pd.DataFrame({
        "km": [1000, 2000],
        "price": [5000, 4000],
    })

    validate_csv(data, ["km", "price"])

def test_validate_csv_missing_columns():
    data = pd.DataFrame({
        "km": [1000, 2000],
    })

    with pytest.raises(ValueError):
        validate_csv(data, ["km", "price"])

def test_validate_csv_empty():
    data = pd.DataFrame(columns=["km", "price"])

    with pytest.raises(ValueError):
        validate_csv(data, ["km", "price"])

def test_validate_csv_non_numeric():
    data = pd.DataFrame({
        "km": [1000, "abc"],
        "price": [5000, 4000],
    })

    with pytest.raises(ValueError):
        validate_csv(data, ["km", "price"])

def test_validate_csv_missing_values():
    data = pd.DataFrame({
        "km": [1000, None],
        "price": [5000, 4000],
    })

    with pytest.raises(ValueError):
        validate_csv(data, ["km", "price"])

def test_validate_csv_infinite_values():
    data = pd.DataFrame({
        "km": [1000, np.inf],
        "price": [5000, 4000],
    })

    with pytest.raises(ValueError):
        validate_csv(data, ["km", "price"])

# Test load_training_data()
def test_load_training_data(tmp_path):
    filepath = tmp_path / "data.csv"

    data = pd.DataFrame({
        "km": [1000, 2000],
        "price": [5000, 4000],
    })

    data.to_csv(filepath, index=False)

    x, y = load_training_data(filepath, "km", "price")

    assert np.array_equal(x, np.array([1000.0, 2000.0]))
    assert np.array_equal(y, np.array([5000.0, 4000.0]))

def test_load_training_data_invalid_csv(tmp_path):
    filepath = tmp_path / "data.csv"

    data = pd.DataFrame({
        "km": [1000, "abc"],
        "price": [5000, 4000],
    })

    data.to_csv(filepath, index=False)

    with pytest.raises(ValueError):
        load_training_data(filepath, "km", "price")

