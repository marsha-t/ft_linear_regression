import argparse
import math
import sys

from linear_regression.model import LinearRegressionModel


def positive_float(value):
    """
    Validate and convert argument into a non-negative finite float

    Args:
        value (str): raw user input

    Returns:
        float: validated mileage

    Raises:
        argparse.ArgumentTypeError:
            If value is not numeric, negative, NaN or infinite
    """
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Mileage must be a number")

    if value < 0:
        raise argparse.ArgumentTypeError("Mileage must be non-negative")
    if not math.isfinite(value):  # covers inf, -inf, nan
        raise argparse.ArgumentTypeError("Mileage must be finite")
    return value


def parse_args():
    """Parse arguments and prompt once if mileage is omitted"""
    parser = argparse.ArgumentParser(description="Predict car price")
    parser.add_argument(
        "--mileage",
        type=positive_float,
        help="Car mileage; prompts if omitted",
    )
    parser.add_argument(
        "--model",
        default="models/model.json",
        help="Path to model JSON",
    )

    args = parser.parse_args()

    if args.mileage is None:
        try:
            args.mileage = positive_float(input("Enter car mileage: "))
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
        except (EOFError, KeyboardInterrupt):
            parser.exit(status=1, message="\nPrediction cancelled.\n")

    return args


def main():
    """
    Parse CLI arguments
    Loads trained regression model
    Predicts car price
    Print predicted car price
    """
    args = parse_args()

    try:
        model = LinearRegressionModel.from_json(args.model)
    except FileNotFoundError:
        print(
            f"Model file not found: {args.model}",
            file=sys.stderr,
        )
        return 1

    prediction = model.predict(args.mileage)

    print(f"Mileage: {args.mileage:,.2f}")
    print(f"Predicted price: {prediction:,.2f}")
    print("Note: Figures are rounded to 2 decimal points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
