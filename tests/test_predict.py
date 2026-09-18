import argparse
from unittest.mock import patch

import pytest

from linear_regression.model import LinearRegressionModel
from predict import main as predict_main
from predict import parse_args as predict_parse_args
from predict import positive_float as predict_positive_float


# Test valid values ###########################################################
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


# Test invalid values #########################################################
@pytest.mark.parametrize(
    "value",
    ["-1", "abc", "inf", "-inf", "nan"],
)
def test_positive_float_invalid(value):
    with pytest.raises(argparse.ArgumentTypeError):
        predict_positive_float(value)


# Test mileage supplied through the command line ##############################
def test_mileage_argument_skips_prompt():
    with (
        patch("sys.argv", ["predict.py", "--mileage", "0"]),
        patch("builtins.input") as mock_input,
    ):
        args = predict_parse_args()

    assert args.mileage == 0
    mock_input.assert_not_called()


# Test interactive mileage input ##############################################
def test_missing_mileage_prompts_once():
    with (
        patch("sys.argv", ["predict.py"]),
        patch("builtins.input", return_value="150000") as mock_input,
    ):
        args = predict_parse_args()

    assert args.mileage == 150000
    mock_input.assert_called_once_with("Enter car mileage: ")


@pytest.mark.parametrize("value", ["abc", "-1", "nan", "inf"])
def test_invalid_prompt_input_exits(value, capsys):
    with (
        patch("sys.argv", ["predict.py"]),
        patch("builtins.input", return_value=value) as mock_input,
    ):
        with pytest.raises(SystemExit) as exc_info:
            predict_parse_args()

    assert exc_info.value.code == 2
    assert "error:" in capsys.readouterr().err
    mock_input.assert_called_once()


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
def test_prompt_cancellation(error, capsys):
    with (
        patch("sys.argv", ["predict.py"]),
        patch("builtins.input", side_effect=error),
    ):
        with pytest.raises(SystemExit) as exc_info:
            predict_parse_args()

    assert exc_info.value.code == 1
    assert "Prediction cancelled." in capsys.readouterr().err


# Test use of selected model
def test_prediction_uses_selected_model(tmp_path, capsys):
    # tmp_path supplies temp directory
    # capsys captures output printed by program
    model_path = tmp_path / "custom_model.json"
    model = LinearRegressionModel(
        theta0=5000,
        theta1=-1000,
        x_mean=100000,
        x_std=50000,
    )
    model.save(model_path)

    with patch(
        "sys.argv",
        [
            "predict.py",
            "--mileage", "150000",
            "--model", str(model_path),
        ],
    ):
        status = predict_main()

    # Retrieve captured stdout and stderr and select stdout
    output = capsys.readouterr().out
    assert status == 0
    assert "Predicted price: 4,000.00" in output


# Test missing model error handling
def test_missing_model_returns_failure(tmp_path, capsys):
    model_path = tmp_path / "missing_model.json"

    with patch(
        "sys.argv",
        [
            "predict.py",
            "--mileage", "100000",
            "--model", str(model_path),
        ],
    ):
        status = predict_main()

    captured = capsys.readouterr()

    assert status == 1
    assert f"Model file not found: {model_path}" in captured.err
    assert "Predicted price:" not in captured.out
