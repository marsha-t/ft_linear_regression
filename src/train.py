import pandas as pd
from model import LinearRegressionModel
from pathlib import Path

def main():
    data_file_path = Path(__file__).resolve().parent.parent / "data" / "data.csv"
    data = pd.read_csv(data_file_path, header=0)
    # TODO CSV Validation

    x = data['km'].tolist() # If x is a Series, x[i] is treated as a label, not int/float
    y = data['price'].tolist()

    model = LinearRegressionModel()
    model.fit(x, y, 0.0000000001, 1000) # TODO remove hardcoded values 
    print(f"Final theta0: {model.theta0:.2f}\nFinal theta1: {model.theta1:.2f}")

    filepath = Path(__file__).resolve().parent.parent / "models" / "theta.json"
    model.save(filepath)

if __name__ == "__main__":
    main()