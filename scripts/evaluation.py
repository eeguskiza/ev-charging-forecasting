"""Rolling-origin (walk-forward) evaluation protocol.

Every notebook (baselines, SARIMA, LSTM, TimesFM) plugs its model in as a
`forecast_fn(history, horizon) -> np.ndarray` callable. The protocol slides
non-overlapping windows of length `horizon` across the test set, feeding the
expanding history to the model at each step. All models therefore see the same
information set and are evaluated on identical actuals.
"""
from __future__ import annotations

from typing import Callable, List, Dict

import numpy as np
import pandas as pd

from .metrics import compute_all


ForecastFn = Callable[[pd.Series, int], np.ndarray]


def rolling_origin_forecast(
    forecast_fn: ForecastFn,
    train: pd.Series,
    test:  pd.Series,
    horizon: int = 7,
) -> List[Dict]:
    """Walk-forward over `test` with non-overlapping windows of length `horizon`.

    For window i:
        history     = train + test[: i * horizon]
        target      = test[i * horizon : (i + 1) * horizon]
        predictions = forecast_fn(history, horizon)
    """
    n_windows = len(test) // horizon
    windows: List[Dict] = []

    for i in range(n_windows):
        cutoff      = i * horizon
        history     = pd.concat([train, test.iloc[:cutoff]])
        target      = test.iloc[cutoff:cutoff + horizon]
        predictions = np.asarray(forecast_fn(history, horizon), dtype=float)

        windows.append({
            'window':      i + 1,
            'origin':      history.index[-1],
            'index':       target.index,
            'actuals':     target.values,
            'predictions': predictions,
        })
    return windows


def evaluate_windows(
    windows: List[Dict],
    y_train: pd.Series,
    season: int = 7,
) -> pd.DataFrame:
    """Compute every metric per window. One row per window."""
    rows = []
    for w in windows:
        row = compute_all(w['actuals'], w['predictions'], y_train.values, season=season)
        row['window'] = w['window']
        row['origin'] = w['origin']
        rows.append(row)
    df = pd.DataFrame(rows)
    cols = ['window', 'origin'] + [c for c in df.columns if c not in ('window', 'origin')]
    return df[cols]


def summarise(per_window: pd.DataFrame) -> pd.Series:
    """Aggregate per-window metrics into mean and std."""
    metric_cols = [c for c in per_window.columns if c not in ('window', 'origin')]
    stats = per_window[metric_cols].agg(['mean', 'std'])
    flat  = {}
    for m in metric_cols:
        flat[f'{m}_mean'] = stats.loc['mean', m]
        flat[f'{m}_std']  = stats.loc['std',  m]
    return pd.Series(flat)


def evaluate_model(
    forecast_fn: ForecastFn,
    train: pd.Series,
    test:  pd.Series,
    horizon: int = 7,
    season:  int = 7,
):
    """End-to-end: rolling-origin forecast + per-window metrics + summary.

    Returns (windows, per_window_df, summary_series).
    """
    windows    = rolling_origin_forecast(forecast_fn, train, test, horizon=horizon)
    per_window = evaluate_windows(windows, train, season=season)
    summary    = summarise(per_window)
    return windows, per_window, summary
