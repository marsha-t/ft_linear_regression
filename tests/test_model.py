import pytest
import json

from linear_regression import LinearRegressionModel

# Test __init__ constructor ###################################################
@pytest.mark.parametrize(
    ("theta0", "theta1"),
    [
        (0, 0),
        (1, 2),
        (-5.5, 3.14),
    ],
)
def test_model_init_valid(theta0, theta1):
    model = LinearRegressionModel(theta0, theta1)

    assert model.theta0 == theta0
    assert model.theta1 == theta1


@pytest.mark.parametrize(
    "theta0",
    ["abc", None, [], {}],
)
def test_model_init_invalid_theta0(theta0):
    with pytest.raises(TypeError):
        LinearRegressionModel(theta0, 1)


@pytest.mark.parametrize(
    "theta1",
    ["abc", None, [], {}],
)
def test_model_init_invalid_theta1(theta1):
    with pytest.raises(TypeError):
        LinearRegressionModel(1, theta1)


@pytest.mark.parametrize(
    "field",
    ["theta0", "theta1", "x_mean", "x_std"],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_init_non_finite_values(field, value):
    parameters = {
        "theta0": 1,
        "theta1": 2,
        "x_mean": 100,
        "x_std": 10,
    }
    parameters[field] = value

    with pytest.raises(ValueError, match="must be finite"):
        LinearRegressionModel(**parameters)


@pytest.mark.parametrize(
    "field",
    ["theta0", "theta1", "x_mean", "x_std"],
)
def test_init_boolean_values(field):
    parameters = {
        "theta0": 1,
        "theta1": 2,
        "x_mean": 100,
        "x_std": 10,
    }
    parameters[field] = True

    with pytest.raises(TypeError, match="must be a number"):
        LinearRegressionModel(**parameters)


def test_init_negative_std():
    with pytest.raises(ValueError, match="strictly positive"):
        LinearRegressionModel(x_mean=100, x_std=-10)


def test_init_missing_x_mean():
    with pytest.raises(ValueError):
        LinearRegressionModel(
            theta0=1,
            theta1=2,
            x_mean=None,
            x_std=10,
        )


def test_init_missing_x_std():
    with pytest.raises(ValueError):
        LinearRegressionModel(
            theta0=1,
            theta1=2,
            x_mean=100,
            x_std=None,
        )


def test_init_zero_std():
    with pytest.raises(ValueError):
        LinearRegressionModel(
            theta0=1,
            theta1=2,
            x_mean=100,
            x_std=0,
        )


# Test fit() ##################################################################
def test_fit_different_lengths():
    model = LinearRegressionModel()

    with pytest.raises(ValueError):
        model.fit([1, 2], [1], 0.01, 100)


def test_fit_empty_data():
    model = LinearRegressionModel()

    with pytest.raises(ValueError):
        model.fit([], [], 0.01, 100)


def test_fit_invalid_learning_rate_type():
    model = LinearRegressionModel()

    with pytest.raises(TypeError):
        model.fit([1], [1], "0.1", 100)


@pytest.mark.parametrize("learning_rate", [0, -1])
def test_fit_invalid_learning_rate_value(learning_rate):
    model = LinearRegressionModel()

    with pytest.raises(ValueError):
        model.fit([1], [1], learning_rate, 100)


@pytest.mark.parametrize(
    "epochs",
    [1.5, "100", None],
)
def test_fit_invalid_epochs_type(epochs):
    model = LinearRegressionModel()

    with pytest.raises(TypeError):
        model.fit([1], [1], 0.01, epochs)


def test_fit_with_scaling_metadata():
    # Raw mileage [90, 110] becomes prepared features [-1, 1]
    model = LinearRegressionModel(x_mean=100, x_std=10)

    model.fit(
        [-1.0, 1.0],
        [8.0, 12.0],
        learning_rate=1.0,
        epochs=1,
        tolerance=0,
    )

    # One batch update learns theta0 = 10 and theta1 = 2.
    # Raw mileage 120 becomes 2, giving a prediction of 14.
    assert model.theta0 == pytest.approx(10)
    assert model.theta1 == pytest.approx(2)
    assert model.predict(120) == pytest.approx(14)


def test_fit_rejects_non_finite_coefficients():
    model = LinearRegressionModel()

    # The proposed intercept is too large for a finite float.
    with pytest.raises(ValueError, match="non-finite coefficients"):
        model.fit(
            [1.0],
            [2.0],
            learning_rate=1e308,
            epochs=1,
        )


def test_fit_rejects_loss_overflow():
    model = LinearRegressionModel()

    # Coefficients remain finite, but squaring the error overflows.
    with pytest.raises(ValueError, match="loss overflowed"):
        model.fit(
            [1.0],
            [1.0],
            learning_rate=1e200,
            epochs=1,
        )


# Test predict() ##############################################################
@pytest.mark.parametrize(
    ("theta0", "theta1", "x", "expected"),
    [
        (0, 1, 5, 5),
        (10, 2, 3, 16),
        (-5, 0.5, 8, -1),
    ],
)
def test_predict(theta0, theta1, x, expected):
    model = LinearRegressionModel(theta0, theta1)

    assert model.predict(x) == expected


@pytest.mark.parametrize(
    "x",
    ["abc", None, [], {}],
)
def test_predict_invalid_x(x):
    model = LinearRegressionModel()

    with pytest.raises(TypeError):
        model.predict(x)


def test_predict_scaled():
    model = LinearRegressionModel(
        theta0=10,
        theta1=2,
        x_mean=100,
        x_std=10,
    )

    # scaled x = (120 - 100) / 10 = 2
    # prediction = 10 + 2 * 2 = 14

    assert model.predict(120) == 14


def test_fit_learns_simple_relationship():
    x = [1, 2, 3, 4]
    y = [2, 4, 6, 8]

    model = LinearRegressionModel()

    model.fit(
        x,
        y,
        learning_rate=0.01,
        epochs=10000,
    )

    prediction = model.predict(5)

    assert prediction == pytest.approx(10, rel=1e-1)


@pytest.mark.parametrize("x", [float("nan"), float("inf"), -float("inf")])
def test_predict_non_finite_input(x):
    model = LinearRegressionModel()

    with pytest.raises(ValueError, match="must be finite"):
        model.predict(x)


def test_predict_initial_model_returns_zero():
    model = LinearRegressionModel()

    assert model.predict(100000) == 0


# Test save() #################################################################
def test_save(tmp_path):
    model = LinearRegressionModel(
        theta0=1,
        theta1=2,
        x_mean=100,
        x_std=10,
    )

    filepath = tmp_path / "model.json"

    model.save(filepath)

    with open(filepath, "r") as file:
        data = json.load(file)

    assert data["theta0"] == 1
    assert data["theta1"] == 2
    assert data["x_mean"] == 100
    assert data["x_std"] == 10


# Test from_json() ############################################################
def test_from_json(tmp_path):
    filepath = tmp_path / "model.json"

    data = {
        "theta0": 1,
        "theta1": 2,
        "x_mean": 100,
        "x_std": 10,
    }

    with open(filepath, "w") as file:
        json.dump(data, file)

    model = LinearRegressionModel.from_json(filepath)

    assert model.theta0 == 1
    assert model.theta1 == 2
    assert model.x_mean == 100
    assert model.x_std == 10


# Test training and persistence together
def test_trained_model_round_trip(tmp_path):
    model = LinearRegressionModel(x_mean=100, x_std=10)
    model.fit(
        [-1.0, 1.0],
        [8.0, 12.0],
        learning_rate=1.0,
        epochs=1,
        tolerance=0,
    )

    mileages = [90, 100, 110, 120]
    predictions_before = [
        model.predict(mileage) for mileage in mileages
    ]

    filepath = tmp_path / "trained_model.json"
    model.save(filepath)
    loaded_model = LinearRegressionModel.from_json(filepath)

    predictions_after = [
        loaded_model.predict(mileage) for mileage in mileages
    ]

    assert predictions_before == pytest.approx([8, 10, 12, 14])
    assert predictions_after == pytest.approx(predictions_before)
