import pytest
import argparse 
from unittest.mock import patch, MagicMock

from src.train import positive_float as train_positive_float
from src.train import positive_int as train_positive_int
from src.train import main as train_main

# Test positive_float()
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("42", 42.0),
        ("42.5", 42.5),
        ("1e3", 1000.0),
        (" 42 ", 42.0)
    ],
)
def test_positive_float_valid(value, expected):
    assert train_positive_float(value) == expected

@pytest.mark.parametrize(
    "value",
    ["-1", "abc", "inf", "-inf", "nan", "0"],
)

def test_positive_float_invalid(value):
    with pytest.raises(argparse.ArgumentTypeError): 
        train_positive_float(value)

# Test positive_int()
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", 1),
        ("100", 100),
    ],
)
def test_positive_int_valid(value, expected):
    assert train_positive_int(value) == expected

@pytest.mark.parametrize(
    "value",
    ["0", "-1", "1.5", "abc"],
)
def test_positive_int_invalid(value):
    with pytest.raises(argparse.ArgumentTypeError):
        train_positive_int(value)
    
# Test train_main()
def test_main_flow():
    with patch("src.train.load_training_data") as mock_load, \
         patch("src.train.standardise") as mock_standardise, \
         patch("src.train.LinearRegressionModel") as mock_model_class, \
         patch("sys.argv", ["train.py"]):

        mock_load.return_value = ([1, 2], [3, 4])

        mock_standardise.return_value = (
            [0.0, 1.0],  # scaled x
            1.5,         # mean
            0.5          # std
        )

        mock_model = MagicMock() # Create fake model object
        mock_model.theta0 = 1234.56
        mock_model.theta1 = 78.9
        mock_model_class.return_value = mock_model

        train_main()

        mock_load.assert_called_once()
        mock_standardise.assert_called_once()
        mock_model.fit.assert_called_once()
        mock_model.save.assert_called_once()

def test_main_standardise_failure(capsys):
    with patch("src.train.load_training_data") as mock_load, \
         patch("src.train.standardise") as mock_standardise, \
         patch("sys.argv", ["train.py"]):

        mock_load.return_value = ([1, 1, 1], [2, 2, 2])

        mock_standardise.side_effect = ValueError(
            "Cannot standardise constant feature"
        )

        train_main()

        captured = capsys.readouterr()

        assert "Cannot standardise constant feature" in captured.out