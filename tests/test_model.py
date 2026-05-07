import pytest
import json

from src import LinearRegressionModel

# Test __init__ constructor
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


# Test fit()
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

# Test predict()
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

def test_predict_missing_x_mean():
    model = LinearRegressionModel(
        theta0=1,
        theta1=2,
        x_mean=None,
        x_std=10,
    )

    with pytest.raises(ValueError):
        model.predict(5)

def test_predict_missing_x_std():
    model = LinearRegressionModel(
        theta0=1,
        theta1=2,
        x_mean=100,
        x_std=None,
    )

    with pytest.raises(ValueError):
        model.predict(5)

def test_predict_zero_std():
    model = LinearRegressionModel(
        theta0=1,
        theta1=2,
        x_mean=100,
        x_std=0,
    )

    with pytest.raises(ValueError):
        model.predict(5)

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

# Test save()
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

# Test from_json()
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