# Full results — all models, all metrics

Every model is evaluated under the same protocol — **rolling-origin walk-forward, 7-day horizon, 4 non-overlapping windows on the last 30 days of the series**. Errors are reported as `mean ± std` across the 4 windows; **MASE** and **R²** are averages. Lower MAE / RMSE / MAPE / MASE / Bias is better; higher R² is better.

For the detailed pipeline (data prep, fits, diagnostics) see the corresponding notebook in `notebooks/`. For the full SoTA-claim breakdown see [`SOTA_ANALYSIS.md`](SOTA_ANALYSIS.md).

## Leaderboard

| Model | MAE (kWh) | RMSE (kWh) | MAPE | MASE | R² | Notebook |
|---|---:|---:|---:|---:|---:|---|
| **Ensemble PatchTST + LSTM (mean)** | **25.09 ± 9.36** | **28.88 ± 11.33** | 58.5% ± 21.6% | **0.63** | -0.13 | `09` |
| **PatchTST (3-seed ensemble)** | **25.19 ± 8.90** | 30.74 ± 11.93 | 58.7% ± 24.6% | 0.64 | -0.32 | `09` |
| ARIMAX (2,1,3) | 25.52 ± 14.38 | 30.51 ± 15.55 | 53.9% ± 29.3% | 0.65 | -0.19 | `08` |
| LSTM seq14_h64_l2 | 25.68 ± 10.64 | 28.47 ± 11.22 | 59.4% ± 24.0% | 0.65 | -0.12 | `05` |
| SARIMAX (2,1,4)x(1,1,2,7) | 25.89 ± 12.76 | 31.12 ± 13.73 | 54.0% ± 25.4% | 0.65 | -0.27 | `04` |
| TimesFM 2.0 — Google (500M, zero-shot) | 26.23 ± 12.59 | 30.79 ± 14.62 | 53.5% ± 21.1% | 0.66 | -0.22 | `06` |
| ARIMA (2,1,3) | 26.31 ± 12.43 | 30.78 ± 13.69 | 58.3% ± 24.2% | 0.67 | -0.25 | `08` |
| TimesFM 1.0 — Google (200M, zero-shot) | 26.32 ± 11.14 | 30.60 ± 12.80 | 57.5% ± 22.8% | 0.67 | -0.25 | `06` |
| AutoARIMA non-seasonal (0,1,1) | 26.40 ± 11.50 | 30.64 ± 12.95 | 57.9% ± 22.0% | 0.67 | -0.25 | `08` |
| ARIMA (3,1,3) | 26.68 ± 11.17 | 31.28 ± 13.28 | 58.2% ± 21.5% | 0.67 | -0.30 | `08` |
| ARMA (3,0,3) | 26.71 ± 12.29 | 30.75 ± 13.91 | 58.7% ± 23.7% | 0.68 | -0.24 | `08` |
| SARIMA (3,1,4)x(0,1,2,7) | 26.86 ± 11.86 | 31.92 ± 13.48 | 57.7% ± 21.9% | 0.68 | -0.36 | `04` |
| Chronos-T5 base — Amazon (200M, zero-shot) | 27.31 ± 11.31 | 32.75 ± 14.11 | 58.7% ± 26.2% | 0.69 | -0.44 | `06` |
| AutoARIMA seasonal (0,1,1)x(1,0,1,7) | 27.44 ± 11.70 | 31.44 ± 13.53 | 57.3% ± 20.9% | 0.69 | -0.31 | `08` |
| Chronos-T5 large — Amazon (710M, zero-shot) | 27.94 ± 13.02 | 33.24 ± 15.97 | 53.6% ± 24.3% | 0.71 | -0.42 | `06` |
| Historical mean *(baseline)* | 30.13 ± 2.18 | 34.87 ± 2.97 | 89.9% ± 29.0% | 0.76 | -1.21 | `03` |
| Last value *(baseline)* | 31.22 ± 12.69 | 37.08 ± 13.78 | 61.0% ± 24.9% | 0.79 | -1.02 | `03` |
| Drift *(baseline)* | 31.24 ± 12.68 | 37.09 ± 13.77 | 61.1% ± 25.0% | 0.79 | -1.03 | `03` |
| ARIMA naive-seasonal (3,1,3) | 34.74 ± 12.09 | 41.36 ± 11.03 | 76.0% ± 24.4% | 0.88 | -1.69 | `08` |
| Seasonal naive *(baseline)* | 34.98 ± 13.14 | 41.57 ± 12.40 | 79.6% ± 25.5% | 0.88 | -1.68 | `03` |

## Model families at a glance

- **New SoTA (notebook 09):** PatchTST trained from scratch (RevIN + mixed exogenous channels + 3-seed ensemble) and its simple mean with the LSTM. Best MAE 25.09 kWh.
- **Classical (S)ARIMA family (notebooks 04, 08):** ARMA, ARIMA, ARIMAX, SARIMA, SARIMAX, AutoARIMA in both seasonal and non-seasonal flavours. Best: ARIMAX (2,1,3), MAE 25.52 kWh.
- **Deep learning (notebook 05):** LSTM with hyper-parameter sweep, best is `seq14_h64_l2`, MAE 25.68 kWh.
- **Foundation models, zero-shot (notebook 06):** Google **TimesFM** 1.0 (200M) + 2.0 (500M) and Amazon **Chronos** T5 base (200M) + T5 large (710M). TimesFM wins the head-to-head; larger Chronos is actually worse.
- **Baselines (notebook 03):** historical mean, last value, drift, seasonal naive, ARIMA naive-seasonal. Top forecasting models reduce MAE by ~28% over the best baseline.

## Notes on the comparison

- **Stacking experiments (Ridge LOWO, NNLS LOWO)** were also run in notebook 09 but are *not* on the leaderboard above because they overfit with only 28 test points. They're documented in the notebook itself.
- **Cross-paper SoTA comparison** (Koohfar 2023, CAT-Former 2025, Kyriakopoulos 2025) is not directly possible because every paper reports in different units, station selections and horizons. See [`SOTA_ANALYSIS.md`](SOTA_ANALYSIS.md) for the full breakdown.
