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
			None or str: None if validation successful, else error messag
	"""
	try: 
		mileage = float(mileage)
	except ValueError:
		return None, "Please input numbers only"
	
	if (mileage < 0):
		return None, "Please input positive numbers only"
	return mileage, None

# Main loop: prompt user for mileage and display prediction
# Loop only ends with correct user input 
def main():
	while True:
		user_input = input("Input mileage: ")
		mileage, error = validate_mileage(user_input)
		if error:
			print(error + "\n")
		else :
			filepath = Path(__file__).resolve().parent.parent / "models" / "theta.json"
			model = LinearRegressionModel.from_json(filepath)
			print(f"Mileage: {mileage:,.2f}")
			print(f"Predicted price: {model.predict(mileage):,.2f}")
			print("Note: Figures are rounded to 2 decimal points")
			break


if __name__ == "__main__":
    main()