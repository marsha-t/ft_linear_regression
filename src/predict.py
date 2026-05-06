from model import LinearRegressionModel
from pathlib import Path
import matplotlib.pyplot as plt 

def validate_mileage(mileage):
	"""
	Validate and convert user input into mileage value

	Args: 
		mileage (str): raw user input

	Returns: 
		tuple:
			float or None: mileage if validation successful, else None
			None or str: None if validation successful, else error message
	"""
	try: 
		mileage = float(mileage)
	except ValueError:
		return None, "Please input numbers only"
	
	if (mileage < 0):
		return None, "Please input positive numbers only"
	return mileage, None

def main():
	"""
	Loads trained model and interactively predicts car price from user-provided input
	Loop only ends with correct user input 
	"""
	filepath = Path(__file__).resolve().parent.parent / "models" / "model.json"
	model = LinearRegressionModel.from_json(filepath)

	while True:
		user_input = input("Input mileage: ")
		mileage, error = validate_mileage(user_input)
		if error:
			print(error + "\n")
		else :
			
			print(f"Mileage: {mileage:,.2f}")
			print(f"Predicted price: {model.predict(mileage):,.2f}")
			print("Note: Figures are rounded to 2 decimal points")
			break


if __name__ == "__main__":
    main()