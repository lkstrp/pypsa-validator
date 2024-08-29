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
    epsilon: float = 1e-9,
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
    epsilon : float, optional (default=1e-9)
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
    y_pred = y_pred + epsilon

    # Calculate the absolute percentage errors
    mape = np.abs((y_true - y_pred) / y_true)

    if aggregate:
        mape = np.mean(mape)

    return mape


def normalized_root_mean_square_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    ignore_inf: bool = True,
    normalization: str = "min-max",
    fill_na: float = 0,
    epsilon: float = 1e-9,
) -> float:
    """
    Calculate the Normalized Root Mean Square Error (NRMSE).

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    ignore_inf : bool, optional (default=True)
        If True, ignore infinite values in the calculation
    normalization : str, optional (default='min-max')
        Method of normalization. Options: 'mean', 'range', 'iqr', 'min-max'
    epsilon : float, optional (default=1e-9)
        Small value to add to normalization factor to avoid division by zero

    Returns
    -------
    float: NRMSE

    """
    y_true_ = np.array(y_true)
    y_pred_ = np.array(y_pred)

    if np.array_equal(y_true_, y_pred_):
        return 0

    # Fill NaN values
    y_true_ = np.nan_to_num(y_true_, nan=fill_na)
    y_pred_ = np.nan_to_num(y_pred_, nan=fill_na)

    # Ignore -inf and inf values
    if ignore_inf:
        mask = np.isfinite(y_true_) & np.isfinite(y_pred_)
        y_true_ = y_true_[mask]
        y_pred_ = y_pred_[mask]

    # Calculate the squared errors
    squared_errors = (y_true_ - y_pred_) ** 2

    # Calculate RMSE
    rmse = np.sqrt(np.mean(squared_errors))

    # Normalize RMSE
    if normalization == "mean":
        normalization_factor = np.mean(y_true_)
    elif normalization == "range":
        normalization_factor = np.ptp(y_true_)
    elif normalization == "iqr":
        q75, q25 = np.percentile(y_true_, [75, 25])
        normalization_factor = q75 - q25
    elif normalization == "min-max":
        normalization_factor = np.max(y_true_) - np.min(y_true_) + epsilon
    else:
        raise ValueError("Invalid normalization method.")

    nrmse = rmse / normalization_factor

    return nrmse
