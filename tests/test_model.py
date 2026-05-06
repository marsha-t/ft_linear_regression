from src import LinearRegressionModel

model = LinearRegressionModel()
x = [1, 2, 3]
y = [2, 4, 6]

model.fit(x, y, 0.1, 20000)

print(model.theta0, model.theta1)
print(model.predict(4))