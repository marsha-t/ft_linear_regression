import argparse
from unittest.mock import patch, MagicMock

import pytest

from train import positive_float as train_positive_float
from train import positive_int as train_positive_int
from train import main as train_main
from linear_regression.model import LinearRegressionModel


# Test positive_float() #######################################################
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


# Test positive_int() #########################################################
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


# Test train_main() ###########################################################
@pytest.mark.parametrize("show_plot", [False, True])
def test_main_flow(show_plot):
    argv = ["train.py"]
    if show_plot:
        argv.append("--plot")

    with patch("train.load_training_data") as mock_load, \
         patch("train.standardise") as mock_standardise, \
         patch("train.LinearRegressionModel") as mock_model_class, \
         patch("train.plot_regression_fit") as mock_plot, \
         patch("train.mse") as mock_mse, \
         patch("train.mae") as mock_mae, \
         patch("train.r2_score") as mock_r2, \
         patch("sys.argv", argv):

        mock_load.return_value = ([1, 2], [3, 4])

        mock_standardise.return_value = (
            [0.0, 1.0],
            1.5,
            0.5
        )

        mock_model = MagicMock()
        mock_model.theta0 = 1234.56
        mock_model.theta1 = 78.9

        mock_model.predict.side_effect = [3.1, 3.9]

        mock_model_class.return_value = mock_model

        mock_mse.return_value = 0.01
        mock_mae.return_value = 0.02
        mock_r2.return_value = 0.99

        status = train_main()

        mock_load.assert_called_once()
        mock_standardise.assert_called_once()

        mock_model.fit.assert_called_once()
        mock_model.predict.assert_any_call(1)
        mock_model.predict.assert_any_call(2)

        mock_mse.assert_called_once()
        mock_mae.assert_called_once()
        mock_r2.assert_called_once()

        assert status == 0

        if show_plot:
            mock_plot.assert_called_once()
        else:
            mock_plot.assert_not_called()

        mock_model.save.assert_called_once()


def test_main_standardise_failure(capsys):
    with (
        patch("train.load_training_data") as mock_load,
        patch("train.standardise") as mock_standardise,
        patch("train.LinearRegressionModel") as mock_model_class,
        patch("sys.argv", ["train.py"]),
    ):
        mock_load.return_value = ([1, 1, 1], [2, 2, 2])
        mock_standardise.side_effect = ValueError(
            "Cannot standardise constant feature"
        )

        status = train_main()

    captured = capsys.readouterr()

    assert status == 1
    assert "Cannot standardise constant feature" in captured.err
    mock_model_class.assert_not_called()


# Test training when R2 is undefined ##########################################
def test_constant_prices_still_save_model(tmp_path, capsys):
    data_path = tmp_path / "constant_prices.csv"
    model_path = tmp_path / "model.json"

    # Mileage varies, but every target price is identical
    data_path.write_text(
        "km,price\n"
        "90,5000\n"
        "100,5000\n"
        "110,5000\n",
        encoding="utf-8",
    )

    with patch(
        "sys.argv",
        [
            "train.py",
            "--data", str(data_path),
            "--output", str(model_path),
        ],
    ):
        status = train_main()

    captured = capsys.readouterr()

    assert status == 0
    assert "R2: unavailable" in captured.out
    assert model_path.is_file()

    model = LinearRegressionModel.from_json(model_path)
    assert model.predict(100) == pytest.approx(5000, abs=1)


# Test failed training preserves the previously saved model ###################
def test_numerical_failure_preserves_model(tmp_path, capsys):
    data_path = tmp_path / "data.csv"
    model_path = tmp_path / "model.json"

    data_path.write_text(
        "km,price\n"
        "90,4000\n"
        "100,5000\n"
        "110,6000\n",
        encoding="utf-8",
    )

    LinearRegressionModel(theta0=123, theta1=0).save(model_path)
    original_contents = model_path.read_bytes()

    with patch(
        "sys.argv",
        [
            "train.py",
            "--data", str(data_path),
            "--output", str(model_path),
            "--lr", "1e100",
            "--epochs", "10",
        ],
    ):
        status = train_main()

    captured = capsys.readouterr()

    assert status == 1
    assert "Training failed:" in captured.err
    assert "try a smaller learning rate" in captured.err
    assert model_path.read_bytes() == original_contents