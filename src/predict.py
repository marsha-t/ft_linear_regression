# Script to run prediction
theta0 = 0 # TODO Temp var
theta1 = 0.01 # TODO Temp var

def validate_mileage(mileage):
	try: 
		mileage = float(mileage)
	except:
		return None, "Please input numbers only"
	
	if (mileage < 0):
		return None, "Please input positive numbers only"
	return mileage, None


def estimate_price(mileage):
	return theta0 + theta1 * mileage

user_input = input("Input mileage: ")
mileage, error = validate_mileage(user_input)
if error:
	print(error)
else :
	print(f"Mileage: {mileage}")
	print(f"Predicted price: {estimate_price(mileage)}")

