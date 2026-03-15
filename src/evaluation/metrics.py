"""
Competition metric: MSE.
"""

from __future__ import annotations

import pandas as pd


def mse(y_true: pd.Series | list, y_pred: pd.Series | list) -> float:
    """Mean Squared Error."""
    if hasattr(y_true, "values"):
        y_true = y_true.values
    if hasattr(y_pred, "values"):
        y_pred = y_pred.values
    return float(((pd.Series(y_true) - pd.Series(y_pred)) ** 2).mean())
