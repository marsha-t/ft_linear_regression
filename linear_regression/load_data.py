import pandas as pd
import numpy as np


def validate_csv(data, required_columns):
    """
    Validate training data

    Args:
        data (pd.DataFrame): training data
        required_columns (list[str]): required column names in dataset

    Returns: None

    Raises:
        ValueError:
            If dataset has missing required columns, no training data,
                non-numeric values, or infinite values
    """
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"CSV missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    if data.empty:
        raise ValueError(
            "CSV contains no training data"
        )

    data = data.loc[:, required_columns].apply(pd.to_numeric, errors='coerce')
    if data.isna().to_numpy().any():
        raise ValueError(
            "CSV values must be numeric and non-empty"
        )

    values = data.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(
            "CSV values must be finite"
        )


def load_training_data(filepath, x_label, y_label):
    """
    load training data with data validated with validate_csv()

    Args:
        filepath (Path | str): path to dataset
        x_label (str): Feature column name
        y_label (str): Target column name

    Returns:
        tuple:
            x (np.ndarray): feature
            y (np.ndarray): target

    Raises:
        ValueError:
            If dataset validation fails
    """
    data = pd.read_csv(filepath, header=0)

    validate_csv(data, [x_label, y_label])

    # Need to convert Series because in pandas, the i in x[i] may be
    #   interpreted as a label rather than position
    # Convert to NumPy array for efficient operations
    x = data[x_label].to_numpy(dtype=float)
    y = data[y_label].to_numpy(dtype=float)

    return x, y
