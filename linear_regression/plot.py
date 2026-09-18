import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def plot_regression_fit(x, y, model):
    """
    Plot scatter plot of mileage and price with fitted regression line from
        given model

    Args:
        x (Sequence(float)): mileage values
        y (Sequence(float)): price values
        model: regression model with predict(value) method

    Returns:
        None: Displays plot in new window
    """
    fig, ax = plt.subplots()

    ax.scatter(x, y, alpha=0.5)

    ax.set_title("Car Mileage and Price")
    ax.set_xlabel("Mileage")
    ax.set_ylabel("Price")

    ax.xaxis.set_major_formatter(
        StrMethodFormatter('{x:,.0f}')
    )

    ax.yaxis.set_major_formatter(
        StrMethodFormatter('${x:,.0f}')
    )

    x_line = [min(x), max(x)]
    y_line = [model.predict(mileage) for mileage in x_line]

    ax.plot(x_line, y_line, color="red", linewidth=2)

    plt.show()
