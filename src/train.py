from model import LinearRegressionModel
from pathlib import Path
from load_data import load_training_data
from preprocessing import standardise

LEARNING_RATE = 0.01
N_ITERS = 1000

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
    print(f"mean(km): {x_mean:.2f}")
    print(f"std(km): {x_std:.2f}\n")

    model = LinearRegressionModel()
    model.fit(x_scaled, y, LEARNING_RATE, N_ITERS)

    model.x_mean = x_mean # values must be set after training, else predict() in fit() will cause double scaling
    model.x_std = x_std

    print("Training complete")
    print(f"theta0: {model.theta0:.2f}\ntheta1: {model.theta1:.2f}\n")

    filepath = Path(__file__).resolve().parent.parent / "models" / "model.json"
    model.save(filepath)
    print(f"Model saved to: models/model.json")


if __name__ == "__main__":
    main()
