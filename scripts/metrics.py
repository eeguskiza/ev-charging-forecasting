"""Forecasting metrics shared across every notebook.

Every metric takes 1-D numpy-like arrays of the same length and returns a float.
`mase` additionally needs the in-sample training series and the seasonal period.
"""
from __future__ import annotations

import numpy as np


def _to_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error, masking zeros in the actuals."""
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    mask = y_true != 0
    if not mask.any():
        return float('nan')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in [0, 2]. Robust to zero actuals."""
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if not mask.any():
        return float('nan')
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def mase(y_true, y_pred, y_train, season: int = 7) -> float:
    """Mean Absolute Scaled Error.

    Scales the MAE on the test window by the in-sample MAE of a seasonal-naive
    forecast on the training series. MASE < 1 beats the trivial naive baseline.
    """
    y_true, y_pred, y_train = _to_array(y_true), _to_array(y_pred), _to_array(y_train)
    if len(y_train) <= season:
        return float('nan')
    naive_errors = np.abs(y_train[season:] - y_train[:-season])
    scale = float(np.mean(naive_errors))
    if scale == 0:
        return float('nan')
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def bias(y_true, y_pred) -> float:
    """Mean signed error. Positive = systematic over-prediction."""
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    return float(np.mean(y_pred - y_true))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination."""
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def compute_all(y_true, y_pred, y_train=None, season: int = 7) -> dict:
    """Return every metric as a dict. `y_train` is required for MASE only."""
    out = {
        'MAE':   mae(y_true, y_pred),
        'RMSE':  rmse(y_true, y_pred),
        'MAPE':  mape(y_true, y_pred),
        'sMAPE': smape(y_true, y_pred),
        'Bias':  bias(y_true, y_pred),
        'R2':    r2(y_true, y_pred),
    }
    if y_train is not None:
        out['MASE'] = mase(y_true, y_pred, y_train, season=season)
    return out
