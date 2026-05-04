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

    # def fit(self, x, y):

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