from pathlib import Path
import matplotlib.pyplot as plt 
import argparse
import math 
from .model import LinearRegressionModel

def positive_float(value):
	"""
	Validate and convert argument into positive finite float

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
		raise argparse.ArgumentTypeError("Mileage must be positive")
	if not math.isfinite(value): # covers inf, -inf, nan
		raise argparse.ArgumentTypeError("Mileage must be finite")
	return value

def main():
	"""
	Parse CLI arguments
	Loads trained regression model
	Predicts car price 
	Print predicted car price 
	"""
	parser = argparse.ArgumentParser(description="Predict car price")
	parser.add_argument("--mileage", type=positive_float, required=True, help="Car mileage")
	parser.add_argument("--model", default="models/model.json", help="Path to trained model JSON")

	args = parser.parse_args()

	filepath = Path(__file__).resolve().parent.parent / "models" / "model.json"
	model = LinearRegressionModel.from_json(filepath)
	print(f"Mileage: {args.mileage:,.2f}")
	print(f"Predicted price: {model.predict(args.mileage):,.2f}")
	print("Note: Figures are rounded to 2 decimal points")

if __name__ == "__main__":
    main()