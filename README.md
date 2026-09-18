# Linear Regression from Scratch

Predicting car prices from mileage with linear regression and batch gradient descent implemented from scratch in Python. Built as part of the [42 curriculum](https://github.com/marsha-t/42-projects), following the [project subject](ft_linear_regression.pdf).

Training and inference are separated: training persists the learned coefficients and feature-scaling statistics as JSON, while prediction loads the persisted model independently.

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
python -m pip install -r requirements.txt
```

## Usage

### Training
Train on the supplied dataset:

```bash
python train.py
```

Training can also be configured and the fitted line visualised:
```bash
python train.py --data data/data.csv  --lr 0.01  --epochs 1000  --output models/model.json  --plot
```

### Prediction

Use the trained model to predict a car's price from its mileage:

```bash
python predict.py --mileage 240000
```

Alternatively, run `python predict.py` to enter the mileage interactively.

A different model file can be selected with --model, and the default model can be restored to its initial zero-coefficient state with:

```bash
python reset_model.py
```

### Tests
Run the tests with:
```
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
## Implementation

The model uses full-batch gradient descent, with mileage standardised before optimisation to improve convergence. Training stops at the epoch limit or when the change in MSE falls below `1e-6`.

The learned coefficients remain in standardised feature space. The persisted JSON therefore stores the coefficients alongside scaling statistics, allowing prediction to reproduce the same preprocessing without access to the training data.

The implementation also validates data, CLI inputs and persisted model state, and prevents failed training runs from overwriting an existing model. Tests cover training, persistence and inference.

## Results

With the supplied dataset and default settings, training produces approximately:

- MAE: 557.83 price units
- R²: 0.733

These are in-sample training metrics, not estimates of performance on unseen cars. The model deliberately uses only mileage and assumes a linear relationship with price; extrapolated predictions are therefore unconstrained and can become unrealistic outside the training range.
