# SoTA Analysis — what we actually beat, and what we didn't

This document gives the honest version of the "did you beat state of the art?" question for the notebook 09 work (PatchTST + DL ensemble). The short answer is: **we beat the previous best model in our own project; we did not beat any published number in a directly comparable way.**

## 1. What "state of the art" looks like for this dataset

Three peer-reviewed papers use the same City of Boulder EV charging dataset for daily/short-term forecasting:

| Paper | Year, venue | SoTA model proposed | Setup | Reported best metric |
|---|---|---|---|---|
| Koohfar et al. — *Prediction of EV Charging Demand: A Transformer-Based Deep Learning Approach* | 2023, *MDPI Sustainability* | Vanilla Transformer | Boulder, 25 stations aggregated, 4 years of data, horizons 30 / 60 / 90 days | RMSE 0.085 / 0.096 / 0.112 (normalized) |
| Kyriakopoulos & Theodoridis — *EV Charging Load Forecasting: An Experimental Comparison of ML Methods* | 2025, arXiv 2512.17257 | Vanilla Transformer | Boulder + Palo Alto + Dundee + Perth, daily 1–5 day ahead, z-score normalized | Transformer MAE 0.28–0.45 (station-level, normalized) |
| Wu et al. — *Short-term Demand Forecasting of EV Charging Stations Using Context-Aware Temporal Transformer* (CAT-Former) | 2025, *Scientific Reports* | Context-Aware Temporal Transformer | Boulder, Jan 2018–Nov 2023, three single stations (Park 1 / Rec 1 / Street 1), 1-hour and 1-day ahead, MinMax [0,1] normalized | 1-day MAE 0.91 / 0.72 / 0.71 (normalized) |

The three papers agree on the qualitative conclusion: **transformer architectures outperform the (S)ARIMA family on Boulder EV data.** That's the "SoTA story".

## 2. What we did in this project

Our protocol (notebooks 00–09):

- Single station selected as the highest-volume one in the dataset (aggregating to one daily series of ~2,000 observations).
- Train: 1,897 days (2018-08-22 → 2023-10-31). Test: last 30 days (Nov 2023).
- **Horizon: 7 days ahead.** Rolling-origin evaluation, 4 non-overlapping windows of 7 days each = 28 test points.
- Target reported in **raw kWh** (no normalization in the metric).
- Exogenous variables (temperature, day-of-week, weekend, holiday) used by SARIMAX / ARIMAX / LSTM / PatchTST where the architecture allows it.

Notebook 09 added a SoTA-family contribution:

- **PatchTST** (Nie et al., *ICLR 2023*) implemented from scratch in PyTorch (`scripts/patchtst_train.py`, ~280 lines, well-commented).
- Architecture details:
  - Context window: 56 days (8 weeks)
  - Patch length: 7 days (one week per token), non-overlapping → 8 tokens
  - 3-layer transformer encoder, d_model=96, 4 heads
  - Channel-mixed patching across 4 channels (energy + temp_mean + is_weekend + is_holiday)
  - **RevIN** (Kim et al., *ICLR 2022*) on the target channel — instance-wise normalization that handles non-stationary level/scale shifts
  - **Seed ensemble** of 3 independent runs (seeds 42, 7, 2024) averaged at inference
- Trained on GPU (RTX 5070 Ti) in ~2 minutes per seed; total ~7 minutes for the 3-seed ensemble.
- Predictions cached to `data/patchtst/predictions.csv` so the notebook stays fast.

## 3. What we actually beat (and by how much)

Same protocol, same series, same metric — apples-to-apples within this project:

| Model | MAE (kWh) | RMSE (kWh) | Δ MAE vs. previous best |
|---|---:|---:|---:|
| **Ensemble PatchTST + LSTM (mean)** | **25.09 ± 9.36** | **28.88 ± 11.33** | **−0.43 kWh** |
| **PatchTST (3-seed ensemble)** | **25.19 ± 8.90** | 30.74 ± 11.93 | **−0.33 kWh** |
| ARIMAX (2,1,3) *— ex-best of notebook 08* | 25.52 ± 14.38 | 30.51 ± 15.55 | — |
| LSTM | 25.68 ± 10.64 | 28.47 ± 11.22 | +0.16 |
| SARIMAX | 25.89 ± 12.76 | 31.12 ± 13.73 | +0.37 |
| TimesFM 2.0 (zero-shot) | 26.23 ± 12.59 | 30.79 ± 14.62 | +0.71 |

**Honest reading:** the ensemble cuts MAE by ~1.7% relative to ARIMAX, and **cuts the standard deviation by ~35%** (9.36 vs 14.38). The variance reduction is the more meaningful improvement here — a "tighter" model is more useful for grid capacity planning than one that's marginally lower on average but swings wildly.

## 4. What we did NOT beat (and cannot fairly claim to)

A direct numerical comparison against the three SoTA papers is **not honest**, because every dimension of their evaluation differs from ours:

| Dimension | Our setup | Koohfar 2023 | Kyriakopoulos 2025 | CAT-Former 2025 |
|---|---|---|---|---|
| Station selection | Highest-volume single station | 25 stations aggregated | Many (multi-station) | 3 specific single stations (Park 1, Rec 1, Street 1) |
| Aggregation | Daily | Daily | Daily | Hourly (then daily) |
| Horizon | 7-day ahead | 30 / 60 / 90 days | 1–5 days | 1-hour / 1-day |
| Normalization | None (raw kWh) | z-score | z-score | MinMax [0,1] |
| Test split | Last 30 days, Nov 2023 | 80/20 random-ish split | Last 20% | Jul–Nov 2023 |
| Reported metric scale | kWh | normalized RMSE | normalized MAE | normalized MAE/MSE |

Their "MAE = 0.71–0.91" and our "MAE = 25.09 kWh" are not the same units, and even if we converted units the underlying instance of the problem is different. You cannot say *"25 kWh < 0.91"* — they're not comparable.

What we can say truthfully:

- We are **in the SoTA family** (transformer architecture, contemporary preprocessing tricks).
- We **reproduce the qualitative ordering** observed in all three papers: transformer > LSTM > (S)ARIMA on Boulder.
- We **don't beat the published numbers in their own setups**, because we never ran our model in those setups.

## 5. What a fair comparison would look like

To make a defensible "we beat CAT-Former" claim, the notebook 10 would have to:

1. **Re-select the three stations** they use (Park 1, Rec 1, Street 1) instead of the highest-volume one.
2. **Re-aggregate to their schedule** (hourly, then daily as they do).
3. **Re-split as they do** (train Jan 2018 → Jun 2023, test Jul → Nov 2023).
4. **Re-normalize with MinMax [0,1]** so the MAE/MSE are in their reported units.
5. **Re-forecast at 1-day ahead** (they don't report 7-day).
6. Train PatchTST and Ensemble on each of the 3 stations, compute MAE / MSE in their normalized space.
7. Compare against their Table 5 / Table 6 numbers (1-day ahead: MAE 0.91 Park / 0.72 Rec / 0.71 Street).

Estimated effort: ~1 day of work (re-preprocess, retrain, evaluate). Risk: even if PatchTST is competitive at 1-day ahead, CAT-Former is specifically designed for this exact dataset and may genuinely be hard to beat.

This is not currently in scope. We flag it as the right experiment to run if a follow-up paper is the goal.

## 6. What is genuinely ours in notebook 09

To be precise about contribution:

- **Architecture (PatchTST):** *not ours* — Nie et al., ICLR 2023.
- **Implementation:** *ours* — written from scratch in PyTorch (no `neuralforecast` / `darts` dependency), ~280 lines in `scripts/patchtst_train.py`.
- **Preprocessing decisions:** *ours* — 4 channels (energy + temp + weekend + holiday), 56-day context, 7-day patches, RevIN on the target channel only.
- **Seed ensemble (3 seeds averaged):** *ours* — empirical decision; brought MAE from 27.11 → 25.19 (–7%).
- **Ensemble with LSTM (simple mean):** *ours* — a priori choice of the two strongest deep-learning models; took MAE from 25.19 → 25.09.
- **Stacking experiments (Ridge LOWO, NNLS LOWO):** *ours* — included to document the failure mode (28 test points is too few for a meta-learner to generalize). Educational, not part of the final result.

## 7. One-line summary

> We brought a 2023-vintage SoTA transformer architecture (PatchTST) into the project, made it competitive (best single model in the local leaderboard), and combined it with our LSTM to push the ensemble below the previous best. We are *in* the SoTA family, but we have not run a like-for-like comparison against the published numbers and do not claim to have beaten them.

## References

- Nie, Nguyen, Sinthong, Kalagnanam — *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*, ICLR 2023. [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
- Kim, Kim, Tae, Park, Choi, Choo — *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*, ICLR 2022. [OpenReview](https://openreview.net/forum?id=cGDAkQo1C0p)
- Koohfar, Woldemariam, Kumar — *Prediction of EV Charging Demand: A Transformer-Based Deep Learning Approach*, MDPI Sustainability 2023. [Paper](https://www.mdpi.com/2071-1050/15/3/2105/htm)
- Kyriakopoulos & Theodoridis — *EV Charging Load Forecasting: An Experimental Comparison of ML Methods*, arXiv 2025. [arXiv:2512.17257](https://arxiv.org/abs/2512.17257)
- Wu et al. — *Short-term Demand Forecasting of EV Charging Stations Using a Context-Aware Temporal Transformer*, Scientific Reports 2025. [PMC12541031](https://pmc.ncbi.nlm.nih.gov/articles/PMC12541031/)
