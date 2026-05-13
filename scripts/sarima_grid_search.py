"""Exhaustive SARIMA / SARIMAX grid search.

Fits every (p, d, q)(P, D, Q)_m combination in the grid over the training
portion of `data/processed_daily.csv` and writes AIC / BIC / log-likelihood
for each model to `data/sarima_grid_results.csv`.

Two variants are searched:
    - SARIMA    (no exogenous variables)
    - SARIMAX   (exog = temp_mean, is_weekend, is_holiday)

Run from the project root or from `scripts/`:

    python scripts/sarima_grid_search.py

Wall time on the MacBook M1 Pro: ~25-35 minutes for the full grid.
"""
from __future__ import annotations

import argparse
import itertools
import os
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

# Resolve paths relative to this file so the script works from any CWD
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
DATA_PATH   = os.path.join(PROJECT_DIR, 'data', 'processed_daily.csv')
OUT_PATH    = os.path.join(PROJECT_DIR, 'data', 'sarima_grid_results.csv')

# Grid configuration (exhaustive, not stepwise)
P_RANGE   = range(0, 5)     # 0..4
Q_RANGE   = range(0, 5)
P_SEAS    = range(0, 3)     # 0..2
Q_SEAS    = range(0, 3)
D_FIXED   = 1
D_SEAS    = 1
SEASON    = 7

TEST_DAYS = 30
EXOG_COLS = ['temp_mean', 'is_weekend', 'is_holiday']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run a tiny grid for smoke-testing.')
    args = parser.parse_args()

    p_range, q_range = (range(0, 2), range(0, 2)) if args.quick else (P_RANGE, Q_RANGE)
    P_range, Q_range = (range(0, 2), range(0, 2)) if args.quick else (P_SEAS,  Q_SEAS)

    data = pd.read_csv(DATA_PATH, parse_dates=['date'], index_col='date')
    split_date = data.index.max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = data.loc[data.index < split_date].copy()

    y    = train['energy_kwh']
    exog = train[EXOG_COLS]

    combos = list(itertools.product(p_range, q_range, P_range, Q_range))
    print(f'Train length        : {len(y)}')
    print(f'Combos per variant  : {len(combos)}')
    print(f'Variants            : SARIMA (no exog), SARIMAX (exog={EXOG_COLS})')

    rows = []
    for variant in ('no_exog', 'exog'):
        ex_train = None if variant == 'no_exog' else exog
        print(f'\n=== Variant: {variant} ===')
        t_variant = time.time()

        for i, (p, q, P, Q) in enumerate(combos, 1):
            order  = (p, D_FIXED, q)
            sorder = (P, D_SEAS,  Q, SEASON)
            t0 = time.time()
            try:
                m = SARIMAX(
                    y, exog=ex_train,
                    order=order, seasonal_order=sorder,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False, maxiter=50)
                aic, bic, llf = float(m.aic), float(m.bic), float(m.llf)
                ok = True
                err = ''
            except Exception as e:
                aic = bic = llf = float('nan')
                ok = False
                err = type(e).__name__

            dt = time.time() - t0
            rows.append({
                'variant': variant,
                'p': p, 'd': D_FIXED, 'q': q,
                'P': P, 'D': D_SEAS, 'Q': Q, 'm': SEASON,
                'aic': aic, 'bic': bic, 'llf': llf,
                'fit_seconds': dt, 'ok': ok, 'error': err,
            })
            if i % 25 == 0 or i == len(combos):
                print(f'  [{i:3d}/{len(combos)}] ({p},1,{q})({P},1,{Q})_{SEASON:d}  '
                      f'AIC={aic:.1f}  fit={dt:.1f}s')

        print(f'Variant {variant} elapsed: {time.time() - t_variant:.0f}s')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f'\nSaved {len(df)} rows -> {OUT_PATH}')

    # Print the top-5 per variant as a quick visual confirmation
    for variant in ('no_exog', 'exog'):
        sub = df[df['variant'] == variant].dropna(subset=['aic']).sort_values('aic').head(5)
        print(f'\nTop-5 by AIC ({variant}):')
        print(sub[['p', 'd', 'q', 'P', 'D', 'Q', 'aic', 'bic']].to_string(index=False))


if __name__ == '__main__':
    main()
