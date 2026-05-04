# Script to run prediction
theta0 = 0 # TODO Temp var
theta1 = 0 # TODO Temp var

def validate_mileage(mileage):
	try: 
		mileage = float(mileage)
	except ValueError:
		return None, "Please input numbers only"
	
	if (mileage < 0):
		return None, "Please input positive numbers only"
	return mileage, None


def estimate_price(mileage, theta0, theta1):
	return theta0 + theta1 * mileage

while True:
	user_input = input("Input mileage: ")
	mileage, error = validate_mileage(user_input)
	if error:
		print(error + "\n")
	else :
		print(f"Mileage: {mileage:,.2f}")
		print(f"Predicted price: {estimate_price(mileage, theta0, theta1):,.2f}")
		print("Note: Figures are rounded to 2 decimal points")
		break

