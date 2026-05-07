from .model import LinearRegressionModel

def main():
    LinearRegressionModel().save("models/model.json")
    print("Model reset")

if __name__ == "__main__":
    main()