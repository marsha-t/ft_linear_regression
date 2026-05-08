import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

def plot_regression_fit(x, y, model):
    """
    Plot scatter plot of mileage and price with fitted regression line from given model

    Args:
        x (Sequence(float)): mileage values
        y (Sequence(float)): price values
        model: regression model with predict(value) method

    Returns:
        None: Displays plot in new window 
        
    """
    plt.scatter(x, y, alpha=0.5)
    plt.title("Car Mileage and Price")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))

    x_line = [min(x), max(x)] 
    y_line = [model.predict(mileage) for mileage in x_line]
    ax.plot(x_line, y_line)
    plt.show()