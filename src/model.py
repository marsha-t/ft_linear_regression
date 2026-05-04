class LinearRegressionModel:
    """
    Simple linear regression model with one feature

    Predicts using y = theta0 + theta1 * x
    """

    def __init__(self, theta0=0, theta1=0):
        """
        Initialises model with given parameters
        
        Args: 
            theta0 (float): intercept (default = 0)
            theta1 (float): slope (default = 0)
        """
        if not isinstance(theta0, (int, float)):
            raise TypeError("theta0 must be a number")
        if not isinstance(theta1, (int, float)):
            raise TypeError("theta1 must be a number")
        
        self.theta0 = theta0
        self.theta1 = theta1

    def fit(self, x, y, learning_rate, n_iters, tolerance=1e-6):
        """
        Train linear regression model using gradient descent

        Args: 
            x (list[float] or array-like): input feature values
            y (list[float] or array-like): target values
            learning_rate (float): step size for gradient descent
            n_iters (int): number of iterations for gradient descent 
            tolerance (float): Min change in mean square error (MSE) required to continue training (default = 0.000001)
                Training stops early if improvement falls below this threshold
                Set to 0 to disable early stopping

        Returns: 
            None: Updates model parameters in place 
        """
        m = len(x)
        if m != len(y):
            raise ValueError("x and y must have same length")
        if m == 0:
            raise ValueError("x and y cannot be empty")
        if not isinstance(learning_rate, (int, float)):
            raise TypeError("learning_rate must be a number")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be a strictly positive number")
        if not isinstance(n_iters, int):
            raise TypeError("n_iters must be an integer")
        if n_iters <= 0:
            raise ValueError("n_iters must be a strictly positive number")
        if not isinstance(tolerance, (int, float)):
            raise TypeError("tolerance must be a number")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        
        prev_loss = float("inf") 

        for _ in range(n_iters): 
            gradient_theta0 = 0 
            gradient_theta1 = 0

            for i in range(m):
                predicted = self.predict(x[i])
                error = predicted - y[i]
                gradient_theta0 += error
                gradient_theta1 += error * x[i]
            
            self.theta0 -= learning_rate * gradient_theta0 / m
            self.theta1 -= learning_rate * gradient_theta1 / m

            # Compute loss after parameter update
            total_squared_error = 0
            for i in range(m):
                predicted = self.predict(x[i])
                error = predicted - y[i]
                total_squared_error += error ** 2
            mean_loss = total_squared_error / m 
            if abs(prev_loss - mean_loss) < tolerance:
                break
            prev_loss = mean_loss 

    def predict(self, x):
        """
        Predict output given single input value

        Args: 
            x (float): input feature value

        Returns:
            float: predicted value 
        """
        if not isinstance(x, (int, float)):
            raise TypeError("x must be a number")
        return self.theta0 + self.theta1 * x