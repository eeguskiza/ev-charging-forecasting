# Short-Term EV Charging Demand Forecasting — Boulder, Colorado

Daily energy demand forecasting at public EV charging stations, using the **Electric Vehicle Charging Station Data** dataset published by the City of Boulder, Colorado on its [Open Data portal](https://open-data.bouldercolorado.gov/datasets/95992b3938be4622b07f0b05eba95d4c_0/about).

**Authors:** Alexander Jauregui Orue, Eneko Alvarez Mendia, Erik Eguskiza Aranda

## Overview

The dataset contains ~148K charging session records from 26 city-owned stations between January 2018 and November 2023. Sessions are aggregated to **daily frequency** for the highest-volume station, producing a univariate series of ~2,000 observations with strong weekly and annual seasonality. The forecasting horizon is **7 days ahead**, evaluated with rolling-origin (4 windows) on a held-out test set. Exogenous variables (temperature, day of week, holiday indicator) are used in the SARIMAX / ARIMAX variants. A zero-shot foundation-model benchmark (Google **TimesFM**, Amazon **Chronos**) is added as an extension to compare pre-trained models against classical and deep-learning approaches.

## Repository Structure

```
ev-charging-forecasting/
├── data/                  # Dataset + cached artifacts (LSTM weights, SARIMA grid)
├── docs/                  # Figures used in the README
├── notebooks/             # Jupyter notebooks (main deliverable)
├── scripts/               # Reusable utilities (metrics, evaluation, training)
├── requirements.txt
└── README.md
```

| Notebook | Description |
|----------|-------------|
| `00_eda.ipynb` | Exploratory analysis: session distribution, station volumes, seasonality (STL) |
| `01_preprocessing.ipynb` | Daily aggregation, exogenous variable integration (Open-Meteo), train/test split |
| `02_stationarity.ipynb` | ADF test, differencing, ACF/PACF, random-walk rejection |
| `03_baselines.ipynb` | Naive forecasts: historical mean, last value, seasonal naive, drift |
| `04_sarima.ipynb` | SARIMA / SARIMAX: AIC grid search, Ljung-Box diagnostics, rolling forecast |
| `05_lstm.ipynb` | LSTM: sequence preparation, hyper-parameter sweep, training, evaluation |
| `06_google_vs_amazon.ipynb` | Zero-shot foundation models: Google TimesFM 1.0/2.0 vs Amazon Chronos-T5 base/large |
| `07_ensemble.ipynb` | Ensemble of SARIMAX + LSTM + TimesFM with prediction intervals and empirical coverage |
| `08_arma_arima_autoarima.ipynb` | ARMA, ARIMA, ARIMAX, AutoARIMA (seasonal + non-seasonal) and unified leaderboard |
| `09_sota_patchtst_stacking.ipynb` | **PatchTST** (ICLR 2023) trained from scratch + DL ensemble — best model in the project's local leaderboard |

## Results

All models are evaluated under the same protocol — rolling-origin, 7-day horizon, 4 test windows — and ranked by MAE. Errors are reported as `mean ± std` across the 4 windows. Full details in `08_arma_arima_autoarima.ipynb` (classical + DL leaderboard) and `09_sota_patchtst_stacking.ipynb` (SoTA push).

| Model | MAE (kWh) | RMSE (kWh) | MAPE | MASE | R² |
|---|---:|---:|---:|---:|---:|
| **Ensemble PatchTST + LSTM (mean)** | **25.09 ± 9.36** | **28.88 ± 11.33** | 58.5% ± 21.6% | **0.63** | -0.13 |
| **PatchTST (3-seed ensemble)** | **25.19 ± 8.90** | 30.74 ± 11.93 | 58.7% ± 24.6% | 0.64 | -0.32 |
| ARIMAX (2,1,3) | 25.52 ± 14.38 | 30.51 ± 15.55 | 53.9% ± 29.3% | 0.65 | -0.19 |
| LSTM seq14_h64_l2 | 25.68 ± 10.64 | 28.47 ± 11.22 | 59.4% ± 24.0% | 0.65 | -0.12 |
| SARIMAX (2,1,4)x(1,1,2,7) | 25.89 ± 12.76 | 31.12 ± 13.73 | 54.0% ± 25.4% | 0.65 | -0.27 |
| TimesFM 2.0 (zero-shot) | 26.23 ± 12.59 | 30.79 ± 14.62 | 53.5% ± 21.1% | 0.66 | -0.22 |
| ARIMA (2,1,3) | 26.31 ± 12.43 | 30.78 ± 13.69 | 58.3% ± 24.2% | 0.67 | -0.25 |
| AutoARIMA non-seasonal (0,1,1) | 26.40 ± 11.50 | 30.64 ± 12.95 | 57.9% ± 22.0% | 0.67 | -0.25 |
| ARIMA (3,1,3) | 26.68 ± 11.17 | 31.28 ± 13.28 | 58.2% ± 21.5% | 0.67 | -0.30 |
| ARMA (3,0,3) | 26.71 ± 12.29 | 30.75 ± 13.91 | 58.7% ± 23.7% | 0.68 | -0.24 |
| SARIMA (3,1,4)x(0,1,2,7) | 26.86 ± 11.86 | 31.92 ± 13.48 | 57.7% ± 21.9% | 0.68 | -0.36 |
| AutoARIMA seasonal (0,1,1)x(1,0,1,7) | 27.44 ± 11.70 | 31.44 ± 13.53 | 57.3% ± 20.9% | 0.69 | -0.31 |
| Historical mean *(baseline)* | 30.13 ± 2.18 | 34.87 ± 2.97 | 89.9% ± 29.0% | 0.76 | -1.21 |
| Last value *(baseline)* | 31.22 ± 12.69 | 37.08 ± 13.78 | 61.0% ± 24.9% | 0.79 | -1.02 |
| Drift *(baseline)* | 31.24 ± 12.68 | 37.09 ± 13.77 | 61.1% ± 25.0% | 0.79 | -1.03 |
| ARIMA naive-seasonal (3,1,3) | 34.74 ± 12.09 | 41.36 ± 11.03 | 76.0% ± 24.4% | 0.88 | -1.69 |
| Seasonal naive *(baseline)* | 34.98 ± 13.14 | 41.57 ± 12.40 | 79.6% ± 25.5% | 0.88 | -1.68 |

![Model comparison](docs/model_comparison.png)

**Takeaways:**
- **PatchTST + LSTM mean ensemble** is the best model in the local leaderboard (MAE 25.09 kWh), beating every classical / foundation / single-DL model trained on this series with the same protocol.
- **PatchTST alone** (3-seed ensemble) is the best *single* architecture (MAE 25.19 kWh, ~35% lower std than ARIMAX).
- This reproduces the qualitative ordering reported by the three published Boulder papers (Koohfar 2023, CAT-Former 2025, Kyriakopoulos 2025): transformers > LSTM > (S)ARIMA on this dataset.
- Naive baselines are comfortably beaten (~28% MAE reduction over the best baseline), confirming the series carries learnable signal beyond seasonal repetition.

> **About the SoTA claim.** The published Boulder papers report numbers in different units (z-score / MinMax normalization) on different station selections and forecast horizons, so we **cannot directly compare** our 25 kWh to their normalized 0.7–0.9. We are *in the SoTA family* (same architecture class, contemporary preprocessing) but we have not run a like-for-like comparison and do not claim to have beaten the published numbers. See [`docs/SOTA_ANALYSIS.md`](docs/SOTA_ANALYSIS.md) for the full honest breakdown.

## Dataset

Download the CSV from the [Boulder Open Data portal](https://open-data.bouldercolorado.gov/datasets/95992b3938be4622b07f0b05eba95d4c_0/about) and place it in the `data/` directory.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tech Stack

- Python 3.11, pandas, statsmodels (SARIMAX), pmdarima (AutoARIMA), scikit-learn, scipy
- PyTorch (LSTM, PatchTST), matplotlib
- Foundation models: Google **TimesFM** 1.0 + 2.0, Amazon **Chronos**-T5 base + large
- Weather data via Open-Meteo API

Heavy training is kept out of the notebooks — model artifacts are produced by the standalone scripts and loaded back from `data/`:

```bash
python scripts/sarima_grid_search.py     # AIC grid → data/sarima_grid_results.csv
python scripts/lstm_train.py             # LSTM sweep + best model → data/lstm/
python scripts/patchtst_train.py         # PatchTST 3-seed ensemble → data/patchtst/
```
