# Short-Term EV Charging Demand Forecasting — Boulder, Colorado

Daily energy demand forecasting at public EV charging stations using the **Electric Vehicle Charging Station Data** dataset, published by the City of Boulder, Colorado through its [Open Data portal](https://open-data.bouldercolorado.gov/datasets/95992b3938be4622b07f0b05eba95d4c_0/about).

**Authors:** Alexander Jauregui Orue, Eneko Alvarez Mendia, Erik Eguskiza Aranda

## Overview

The dataset contains ~148K charging session records from 26 city-owned stations spanning January 2018 to November 2023. Each record includes station name, timestamps, charging duration, energy consumed (kWh), and GHG savings. The data is aggregated to **daily frequency** for the highest-volume station, producing a univariate time series of ~2,000 observations with strong weekly and annual seasonality.

The forecasting horizon is **7 days ahead**, motivated by the practical need for weekly grid capacity planning. External exogenous variables (temperature, day of week, holiday indicator) are incorporated for SARIMAX modelling.

As an extension, a zero-shot foundation-model benchmark is added using Google's **TimesFM** to compare a pre-trained model against the classical and deep-learning approaches.

## Repository Structure

```
ev-charging-forecasting/
├── data/                  # Dataset (not tracked by git)
├── notebooks/             # Jupyter notebooks (main deliverable)
│   ├── 00_eda.ipynb
│   ├── 01_preprocessing.ipynb
│   ├── 02_stationarity.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_sarima.ipynb
│   ├── 05_lstm.ipynb
│   └── 06_timesfm.ipynb
├── scripts/               # Utility scripts
├── .gitignore
└── README.md
```

## Dataset

Download the CSV from the [Boulder Open Data portal](https://open-data.bouldercolorado.gov/datasets/95992b3938be4622b07f0b05eba95d4c_0/about) and place it in the `data/` directory.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `00_eda.ipynb` | Exploratory analysis: session distribution, station volumes, seasonality patterns |
| `01_preprocessing.ipynb` | Daily aggregation, exogenous variable integration, train/test split |
| `02_stationarity.ipynb` | ADF test, differencing, ACF/PACF analysis, random walk rejection |
| `03_baselines.ipynb` | Naive forecasts: historical mean, last value, seasonal baseline (MAPE) |
| `04_sarima.ipynb` | SARIMA/SARIMAX: AIC grid search, residual diagnostics (Ljung-Box), forecasting |
| `05_lstm.ipynb` | LSTM model: sequence preparation, training, comparison with SARIMA |
| `06_timesfm.ipynb` | *Extra:* zero-shot forecast with Google TimesFM (foundation model), benchmarked against SARIMA and LSTM |

## Tech Stack

- Python 3.11, pandas, statsmodels (SARIMAX), scikit-learn
- PyTorch (LSTM), matplotlib
- TimesFM (`timesfm[torch]`, HF checkpoint `google/timesfm-2.0-500m-pytorch`)
- Weather data via Open-Meteo API
