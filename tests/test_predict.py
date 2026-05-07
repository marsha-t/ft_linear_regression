from src import positive_float

import argparse
import pytest

def test_positive_float_valid():
	assert(positive_float("42") == 42.0)

def test_positive_float_decimal():
	assert(positive_float("42.5") == 42.5)

def test_positive_float_negative():
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float("-1")
        
def test_positive_float_non_number():
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float("abc")
        
def test_positive_float_inf():
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float("inf")
        
def test_positive_float_nan():
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float("nan")
        

# @pytest.mark.parametrize(
#     "value",
#     ["-1", "abc", "inf", "-inf", "nan"]
# )
# def test_positive_float_invalid(value):
#     with pytest.raises(argparse.ArgumentTypeError):
#         positive_float(value)