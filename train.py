import argparse
import math
import sys
from pandas.errors import EmptyDataError, ParserError

from linear_regression.load_data import load_training_data
from linear_regression.preprocessing import standardise
from linear_regression.model import LinearRegressionModel
from linear_regression.plot import plot_regression_fit
from linear_regression.metrics import mse, mae, r2_score


def positive_float(value):
    """
    Validate and convert argument into positive finite float

    Args:
        value (str): raw user input

    Returns:
        float: validated learning rate

    Raises:
        argparse.ArgumentTypeError:
            If value is not numeric, negative, NaN or infinite
    """
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Learning rate must be a number"
        )

    if value <= 0:
        raise argparse.ArgumentTypeError(
            "Learning rate must be positive"
        )

    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(
            "Learning rate must be finite"
        )

    return value


def positive_int(value):
    """
    Validate and convert argument into positive integer

    Args:
        value (str): raw user input

    Returns:
        int: validated number of epochs

    Raises:
        argparse.ArgumentTypeError:
            If value is not a positive integer
    """
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Epochs must be an integer"
        )

    if value <= 0:
        raise argparse.ArgumentTypeError(
            "Epochs must be positive"
        )

    return value


def parse_args():
    """Parse and validate training command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Train linear regression model"
    )
    parser.add_argument(
        "--data",
        default="data/data.csv",
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--lr",
        default=0.01,
        type=positive_float,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        default=1000,
        type=positive_int,
        help="Number of epochs",
    )
    parser.add_argument(
        "--output",
        default="models/model.json",
        help="Path to model JSON",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display the data and fitted regression line after saving",
    )
    return parser.parse_args()


def train(args):
    """
    Load training data, standardise features, train linear regression model
        and save parameters to JSON
    """
    x, y = load_training_data(args.data, "km", "price")
    print(f"Loaded dataset: {len(y)} samples\n")

    x_scaled, x_mean, x_std = standardise(x)

    print("Feature standardisation:")
    print(f"mean(km): {x_mean:,.2f}")
    print(f"std(km): {x_std:,.2f}\n")

    model = LinearRegressionModel(
        x_mean=x_mean,
        x_std=x_std,
    )
    model.fit(x_scaled, y, args.lr, args.epochs)

    print("Training complete")
    print(f"theta0: {model.theta0:,.2f}")
    print(f"theta1: {model.theta1:,.2f}\n")

    predictions = [model.predict(mileage) for mileage in x]
    mse_value = mse(y, predictions)
    mae_value = mae(y, predictions)

    try:
        r2_value = r2_score(y, predictions)
    except ValueError:
        r2_value = None

    print("Training metrics")
    print(f"Mean Squared Error: {mse_value:,.2f}")
    print(f"Mean Absolute Error: {mae_value:,.2f}")
    if r2_value is None:
        print("R2: unavailable (all target prices are identical)\n")
    else:
        print(f"R2: {r2_value:,.2f}\n")

    model.save(args.output)
    print(f"Model saved to: {args.output}")

    if args.plot:
        plot_regression_fit(x, y, model)


def main():
    """Handle command-line arguments and expected training errors."""
    args = parse_args()

    try:
        train(args)
    except (OSError, ValueError, EmptyDataError, ParserError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
