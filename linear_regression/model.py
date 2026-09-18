import json
import math


class LinearRegressionModel:
    """
    Simple linear regression model with one feature

    Predicts using y = theta0 + theta1 * x
    """

    @staticmethod
    def _validate_number(name, value):
        """Require a finite numeric value, excluding booleans."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

    def __init__(self, theta0=0, theta1=0, x_mean=None, x_std=None):
        """
        Initialises model with given parameters

        Args:
            theta0 (float): intercept (default = 0)
            theta1 (float): slope (default = 0)
            x_mean (float | None): mean used for standardisation
            x_std (float | None): standard deviation used for standardisation

        Raises:
            TypeError:
                If coefficients or scaling statistics are not numbers
            ValueError:
                If values are non-finite, scaling statistics are incomplete,
                or standard deviation is not strictly positive.
        """
        self._validate_number("theta0", theta0)
        self._validate_number("theta1", theta1)
        if (x_mean is None) != (x_std is None):
            raise ValueError(
                "x_mean and x_std must either both be provided or both be None"
            )
        if x_mean is not None:
            self._validate_number("x_mean", x_mean)
            self._validate_number("x_std", x_std)
            if x_std <= 0:
                raise ValueError("x_std must be strictly positive")

        self.theta0 = theta0
        self.theta1 = theta1
        self.x_mean = x_mean
        self.x_std = x_std

    def _predict_prepared(self, x):
        """Evaluate the linear equation on an
            already-prepared/standardised feature"""
        return self.theta0 + self.theta1 * x

    def fit(self, x, y, learning_rate, epochs, tolerance=1e-6):
        """
        Train linear regression model using gradient descent

        Args:
            x (list[float] or array-like): Prepared/Standardised feature values
                If the model stores scaling statistics, these values must
                already be standardised using those statistics. fit() does not
                apply scaling.
            y (list[float] or array-like): target values
            learning_rate (float): step size for gradient descent
            epochs (int): number of epochs for gradient descent
            tolerance (float): Min change in mean square error (MSE) required
                to continue training (default = 0.000001).
                Training stops when absolute change in MSE falls below this
                    threshold.
                Set to 0 to disable early stopping.

        Returns:
            None: Updates model parameters in place

        Raises:
            ValueError:
                - x and y have different lengths or are empty
                - learning_rate is not strictly positive
                - epochs is not strictly positive
                - tolerance is negative
                - coefficient updates or training loss become non-finite
                - the training loss calculation overflows

            TypeError:
                - learning_rate is not numeric
                - epochs is not an integer
                - tolerance is not numeric
        """
        m = len(x)
        if m != len(y):
            raise ValueError("x and y must have same length")
        if m == 0:
            raise ValueError("x and y cannot be empty")
        if not isinstance(learning_rate, (int, float)):
            raise TypeError("learning_rate must be a number")
        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be a strictly positive number"
            )
        if not isinstance(epochs, int):
            raise TypeError("epochs must be an integer")
        if epochs <= 0:
            raise ValueError("epochs must be a strictly positive number")
        if not isinstance(tolerance, (int, float)):
            raise TypeError("tolerance must be a number")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")

        prev_loss = float("inf")

        for _ in range(epochs):
            gradient_theta0 = 0
            gradient_theta1 = 0

            for i in range(m):
                predicted = self._predict_prepared(x[i])
                error = predicted - y[i]
                gradient_theta0 += error
                gradient_theta1 += error * x[i]

            next_theta0 = self.theta0 - learning_rate * gradient_theta0 / m
            next_theta1 = self.theta1 - learning_rate * gradient_theta1 / m

            if not math.isfinite(
                next_theta0
            ) or not math.isfinite(
                next_theta1
            ):
                raise ValueError(
                    "Training produced non-finite coefficients; "
                    "try a smaller learning rate"
                )

            self.theta0 = next_theta0
            self.theta1 = next_theta1

            # Compute loss after parameter update
            total_squared_error = 0
            try:
                for i in range(m):
                    predicted = self._predict_prepared(x[i])
                    error = predicted - y[i]
                    total_squared_error += error ** 2

                mean_loss = total_squared_error / m
            except OverflowError as error:
                raise ValueError(
                    "Training loss overflowed; try a smaller learning rate"
                ) from error

            if not math.isfinite(mean_loss):
                raise ValueError(
                    "Training produced non-finite loss; "
                    "try a smaller learning rate"
                )

            if abs(prev_loss - mean_loss) < tolerance:
                break
            prev_loss = mean_loss

    def predict(self, x):
        """
        Predict output given single input value
        If scaling parameters are available, input is standardised
            using stored mean and standard deviation

        Args:
            x (float): input feature value

        Returns:
            float: predicted value

        Raises:
            TypeError:
                If x is not a number
            ValueError:
                If model scaling parameters are incomplete or invalid
        """
        self._validate_number("x", x)
        if self.x_mean is not None or self.x_std is not None:
            if self.x_mean is None or self.x_std is None:
                raise ValueError("Model scaling parameters are invalid")
            if self.x_std == 0:
                raise ValueError("Model scaling parameters are invalid")
            x = (x - self.x_mean) / self.x_std
            self._validate_number("standardised x", x)
        return self._predict_prepared(x)

    def save(self, filepath):
        """
        Save model parameters to JSON file

        Args:
            filepath (str): filepath of JSON

        Returns:
            None
        """
        data = {
            "theta0": self.theta0,
            "theta1": self.theta1,
            "x_mean": self.x_mean,
            "x_std": self.x_std
        }
        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def from_json(cls, filepath):
        """
        Loads model parameters from JSON file and creates new instance with
            loaded parameters

        Args:
            filepath (str): filepath of JSON

        Returns:
            cls: new instance initialised with loaded parameters
        """
        with open(filepath, "r") as file:
            data = json.load(file)
        return cls(
            data["theta0"], data["theta1"], data["x_mean"], data["x_std"]
        )
