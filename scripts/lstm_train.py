"""Train an LSTM forecaster on the daily EV charging series.

Sweeps a small hyperparameter grid, keeps the model with the best validation
loss, and persists weights + normalisation stats + sweep history to
`data/lstm/`. The notebook `05_lstm.ipynb` loads those artefacts and runs the
rolling-origin evaluation — no training happens inside the notebook.

Device auto-detection: cuda → mps → cpu. The expensive runs (full sweep) are
meant for the RTX 5070 Ti; the script still works on M1 Pro (MPS) but slower.

Usage:
    python scripts/lstm_train.py
"""
from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent
DATA_PATH   = PROJECT_DIR / 'data' / 'processed_daily.csv'
OUT_DIR     = PROJECT_DIR / 'data' / 'lstm'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_DAYS    = 30        # held out for the rolling-origin evaluation in the notebook
VAL_DAYS     = 60        # held out from train for hyperparameter selection
TARGET_COL   = 'energy_kwh'
FEATURE_COLS = ['energy_kwh', 'temp_mean', 'is_weekend', 'is_holiday', 'month']

# Hyperparameter sweep
SEQ_LENS     = [14, 28]
HIDDEN_SIZES = [32, 64]
NUM_LAYERS   = [1, 2]
DROPOUT      = 0.1
LR           = 1e-3
BATCH_SIZE   = 64
MAX_EPOCHS   = 100
PATIENCE     = 12
SEED         = 42


def get_device() -> torch.device:
    """Auto-select cuda → mps → cpu."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class LSTMForecaster(nn.Module):
    """One-step-ahead LSTM. Multi-step at inference time is autoregressive."""

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def make_sequences(df: pd.DataFrame, seq_len: int):
    """Build (X, y) windows. X: (N, seq_len, n_features), y: (N,) next-day energy."""
    arr        = df[FEATURE_COLS].values.astype(np.float32)
    target_idx = FEATURE_COLS.index(TARGET_COL)

    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i : i + seq_len])
        y.append(arr[i + seq_len, target_idx])
    return np.asarray(X), np.asarray(y)


def standardise(X, mean=None, std=None):
    """Z-score normalisation per feature column."""
    if mean is None:
        flat = X.reshape(-1, X.shape[-1])
        mean = flat.mean(axis=0)
        std  = flat.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


# --------------------------------------------------------------------------- #
# Single-config training loop
# --------------------------------------------------------------------------- #
def train_config(seq_len, hidden_size, num_layers, train_df, val_df, device):
    X_train, y_train = make_sequences(train_df, seq_len)
    X_val,   y_val   = make_sequences(val_df,   seq_len)

    X_train_n, X_mean, X_std = standardise(X_train)
    X_val_n,   _, _          = standardise(X_val, X_mean, X_std)

    y_mean = float(y_train.mean())
    y_std  = float(y_train.std() + 1e-8)
    y_train_n = (y_train - y_mean) / y_std
    y_val_n   = (y_val   - y_mean) / y_std

    tr_ds  = TensorDataset(torch.from_numpy(X_train_n), torch.from_numpy(y_train_n))
    val_ds = TensorDataset(torch.from_numpy(X_val_n),   torch.from_numpy(y_val_n))
    tr_dl  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = LSTMForecaster(len(FEATURE_COLS), hidden_size, num_layers, DROPOUT).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()

    best_val_loss = float('inf')
    best_state    = None
    bad_epochs    = 0
    history       = []

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        tr_losses = []
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(crit(model(xb), yb).item())

        tr_loss  = float(np.mean(tr_losses))
        val_loss = float(np.mean(val_losses))
        history.append({'epoch': epoch, 'train_loss': tr_loss, 'val_loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs    = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    norms = {
        'X_mean': X_mean.tolist(),
        'X_std':  X_std.tolist(),
        'y_mean': y_mean,
        'y_std':  y_std,
    }
    return {
        'best_val_loss': best_val_loss,
        'state':         best_state,
        'history':       history,
        'epochs':        epoch,
        'norms':         norms,
    }


# --------------------------------------------------------------------------- #
# Main: sweep over hyperparameters, persist the winner
# --------------------------------------------------------------------------- #
def main():
    device = get_device()
    print(f'Device: {device}')
    torch.manual_seed(SEED)

    data = pd.read_csv(DATA_PATH, parse_dates=['date'], index_col='date')

    split_date  = data.index.max() - pd.Timedelta(days=TEST_DAYS - 1)
    train_full  = data.loc[data.index < split_date].copy()

    val_cut     = train_full.index.max() - pd.Timedelta(days=VAL_DAYS - 1)
    train_df    = train_full.loc[train_full.index <  val_cut].copy()
    val_df      = train_full.loc[train_full.index >= val_cut].copy()

    print(f'Train: {len(train_df)}   Val: {len(val_df)}   Features: {FEATURE_COLS}')

    configs = list(itertools.product(SEQ_LENS, HIDDEN_SIZES, NUM_LAYERS))
    print(f'Configs to train: {len(configs)}\n')

    sweep_rows = []
    best       = {'val_loss': float('inf')}

    for seq_len, hidden, layers in configs:
        name = f'seq{seq_len}_h{hidden}_l{layers}'
        print(f'--- {name} ---')
        t0     = time.time()
        result = train_config(seq_len, hidden, layers, train_df, val_df, device)
        dt     = time.time() - t0
        print(f'  val_loss={result["best_val_loss"]:.4f}  '
              f'epochs={result["epochs"]}  time={dt:.1f}s')

        sweep_rows.append({
            'name':          name,
            'seq_len':       seq_len,
            'hidden_size':   hidden,
            'num_layers':    layers,
            'dropout':       DROPOUT,
            'best_val_loss': result['best_val_loss'],
            'epochs':        result['epochs'],
            'fit_seconds':   dt,
        })

        if result['best_val_loss'] < best['val_loss']:
            best = {
                'val_loss': result['best_val_loss'],
                'name':     name,
                'config':   {'seq_len': seq_len, 'hidden_size': hidden,
                             'num_layers': layers, 'dropout': DROPOUT},
                'state':    result['state'],
                'norms':    result['norms'],
                'history':  result['history'],
            }

    # Persist the winner and the sweep table
    torch.save(best['state'], OUT_DIR / 'best_model.pt')
    with open(OUT_DIR / 'best_config.json', 'w') as f:
        json.dump({
            'name':         best['name'],
            'config':       best['config'],
            'norms':        best['norms'],
            'feature_cols': FEATURE_COLS,
            'target_col':   TARGET_COL,
            'val_loss':     best['val_loss'],
        }, f, indent=2)
    pd.DataFrame(sweep_rows).to_csv(OUT_DIR / 'sweep_results.csv', index=False)
    pd.DataFrame(best['history']).to_csv(OUT_DIR / 'best_history.csv', index=False)

    print(f'\nBest config: {best["name"]}   val_loss={best["val_loss"]:.4f}')
    print(f'Artefacts saved to: {OUT_DIR}/')


if __name__ == '__main__':
    main()
