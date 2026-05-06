import numpy as np
from model import LinearRegressionModel
from pathlib import Path
from load_data import load_training_data

def standardise(values):
    """
    Standardise values using z-score normalisation
    Converts input to NumPy array, then scales data to mean 0 and std 1
    Conversion is needed for vector arithmetic

    Args:
        values (np.ndarray): sequence of values to standardise

    Returns:
        tuple:
            scaled (np.ndarray): standardised values
            mean (float): mean of original values
            std (float): standard deviation of original values

    Raises:
        ValueError:
            If standard deviation is zero
    """
    values = np.array(values, dtype=float)
    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        raise ValueError("Cannot standardise values with zero standard deviation")
    scaled = (values - mean) / std
    return scaled, mean, std

def main():
    """
    Load training data, standardise features, train linear regression model and save parameters to JSON
    """
    data_file_path = Path(__file__).resolve().parent.parent / "data" / "data.csv"
    
    x, y = load_training_data(data_file_path, "km", "price")
    print(f"Loaded dataset: {len(y)} samples\n")

    try:
        x_scaled, x_mean, x_std  = standardise(x)
    except ValueError as error:
        print(error)
        return
    print("Feature standardisation:")
    print(f"mean(km): {x_mean}")
    print(f"std(km): {x_std}\n")

    model = LinearRegressionModel()
    model.fit(x_scaled, y, 0.01, 1000) # TODO remove hardcoded values

    model.x_mean = x_mean # values must be set after training, else predict() in fit() will cause double scaling
    model.x_std = x_std

    print("Training complete")
    print(f"theta0: {model.theta0:.2f}\ntheta1: {model.theta1:.2f}\n")

    filepath = Path(__file__).resolve().parent.parent / "models" / "model.json"
    model.save(filepath)
    print(f"Model saved to: models/model.json")


if __name__ == "__main__":
    main()
