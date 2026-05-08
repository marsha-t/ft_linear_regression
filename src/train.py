import argparse
import math

from .load_data import load_training_data
from .preprocessing import standardise
from .model import LinearRegressionModel
from .plot import plot_regression_fit

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

def main():
    """
    Load training data, standardise features, train linear regression model and save parameters to JSON
    """
    parser = argparse.ArgumentParser(description="Train linear regression model")
    parser.add_argument("--data", default="data/data.csv", help="Path to CSV dataset")
    parser.add_argument("--lr", default=0.01, type=positive_float, help="Learning rate")
    parser.add_argument("--epochs", default=1000, type=positive_int, help="Number of epochs")
    parser.add_argument("--output", default="models/model.json", help="Path to model JSON")

    args = parser.parse_args()

    x, y = load_training_data(args.data, "km", "price")
    print(f"Loaded dataset: {len(y)} samples\n")

    try:
        x_scaled, x_mean, x_std  = standardise(x)
    except ValueError as error:
        print(error)
        return
    print("Feature standardisation:")
    print(f"mean(km): {x_mean:.2f}")
    print(f"std(km): {x_std:.2f}\n")

    model = LinearRegressionModel()
    model.fit(x_scaled, y, args.lr, args.epochs)

    model.x_mean = x_mean # values must be set after training, else predict() in fit() will cause double scaling
    model.x_std = x_std

    print("Training complete")
    print(f"theta0: {model.theta0:.2f}")
    print(f"theta1: {model.theta1:.2f}\n")

    plot_regression_fit(x, y, model)
    
    model.save(args.output)
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
