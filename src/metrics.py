import numpy as np 

def mse(y_actual, y_pred):
    """
    Calculate Mean Squared Error (MSE) between actual and predicted values

    Args:
        y_actual (np.ndarray): actual target values
        y_pred (np.ndarray): predicted target values
    
    Returns:
        float: mean squared error value
    """
    return np.mean((y_actual - y_pred) ** 2)

def mae(y_actual, y_pred):
    """
    Calculate Mean Absolute Error (MAE) between actual and predicted values
    
    Args:
        y_actual (np.ndarray): actual target values
        y_pred (np.ndarray): predicted target values
    
    Returns:
        float: mean absolute error value
    """
    return np.mean(np.abs(y_actual - y_pred))

def r2_score(y_actual, y_pred):
    """
    Calculate R-squared (R2) score between actual and predicted values
    
    Args:
        y_actual (np.ndarray): actual target values
        y_pred (np.ndarray): predicted target values
    
    Returns:
        float: R-squared value

    Raises:
        ValueError: If all actual values are the same
    """
    tss = np.sum((y_actual - np.mean(y_actual)) ** 2)
    rss = np.sum((y_actual - y_pred) ** 2)
    if tss == 0:
        raise ValueError("R2 is undefined when all actual values are the same")
    return 1 - (rss / tss)
