import argparse
import pytest
from unittest.mock import patch

from src import predict_positive_float, predict_main

# Test valid values
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("42", 42.0),
        ("42.5", 42.5),
        ("0", 0.0),
        ("1e3", 1000.0),
        (" 42 ", 42.0)
    ],
)
def test_positive_float_valid(value, expected):
    assert predict_positive_float(value) == expected

# Test invalid values
@pytest.mark.parametrize(
    "value",
    ["-1", "abc", "inf", "-inf", "nan"],
)
def test_positive_float_invalid(value):
    with pytest.raises(argparse.ArgumentTypeError): 
        predict_positive_float(value)

# Test missing required argument
def test_missing_mileage():
    with patch("sys.argv", ["predict.py"]): # temporary replace sys.argv (Python CLI): run python predict.py
        with pytest.raises(SystemExit) as exc_info: 
            predict_main()

    assert exc_info.value.code == 2