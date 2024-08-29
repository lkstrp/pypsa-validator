"""
Helper module for calculating evaluation metrics.
"""


import numpy as np
from numpy.typing import ArrayLike


def min_max_normalized_mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculate the min-max normalized Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : array-like
        True values

    y_pred : array-like
        Predicted values

    Returns
    -------
    float: Min-max normalized MAE

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ignore -inf and inf values in y_true
    y_true = y_true[np.isfinite(y_true)]
    y_pred = y_pred[np.isfinite(y_pred)]

    # Calculate the absolute errors
    abs_errors = np.abs(y_true - y_pred)

    # Check if all errors are the same
    if np.all(abs_errors == abs_errors[0]):
        return 0  # Return 0 if all errors are identical to avoid division by zero

    # Min-max normalization
    min_error = np.min(abs_errors)
    max_error = np.max(abs_errors)

    normalized_errors = (abs_errors - min_error) / (max_error - min_error)

    return np.mean(normalized_errors)


def mean_absolute_percentage_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    epsilon: float = 1e-5,
    aggregate: bool = True,
    ignore_inf=True,
) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE).

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    epsilon : float, optional (default=1e-10)
        Small value to avoid division by zero

    Returns
    -------
    float: MAPE

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ignore -inf and inf values
    if ignore_inf:
        y_true = y_true[np.isfinite(y_true)]
        y_pred = y_pred[np.isfinite(y_pred)]

    # Avoid division by zero
    y_true = y_true + epsilon
    y_pred = y_pred + epsilon

    # Calculate the absolute percentage errors
    mape = np.abs((y_true - y_pred) / y_true)

    if aggregate:
        mape = np.mean(mape)

    return mape

