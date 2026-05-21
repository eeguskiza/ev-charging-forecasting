"""Train a PatchTST forecaster on the daily EV charging series.

PatchTST (Nie et al., ICLR 2023) splits a univariate series into fixed-length
patches, embeds them as transformer tokens, and produces the full forecast
horizon in one direct linear projection (no autoregression). It currently
holds state-of-the-art on multiple time-series benchmarks (ETT, Weather,
Electricity) and is the closest "open" architecture to the transformer
variants used in the Boulder EV forecasting papers (Koohfar 2023, CAT-Former
2025).

We implement it from scratch in PyTorch — no neuralforecast / darts dependency.

Pipeline
--------
1. Read `data/processed_daily.csv`, hold out the last 30 days as test, the
   prior 60 days as validation.
2. Standardise the energy series with z-score stats computed on `train` only.
3. Train PatchTST(context_len=56, patch_len=7, stride=7) → direct 7-step head.
   Early-stop on validation MSE, save the best weights.
4. Run rolling-origin inference: for each of the 4 non-overlapping windows in
   the test set, take the last 56 days of the *expanding* history and produce
   the 7-day forecast. Predictions are inverse-normalised and persisted to
   `data/patchtst/predictions.csv`, one row per (window, day).

The notebook `09_sota_patchtst_stacking.ipynb` loads that CSV and plugs it
straight into the standard `evaluate_model()` utility — no training inside
the notebook.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
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
DATA_PATH   = PROJECT_DIR / "data" / "processed_daily.csv"
OUT_DIR     = PROJECT_DIR / "data" / "patchtst"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL   = "energy_kwh"
# Exogenous channels appended to the target. PatchTST will see all of them
# through a channel-mixed patch embedding; the head still predicts only the
# target. Day-of-week, weekend/holiday flags and temperature are all known a
# week ahead (calendar + standard weather forecast), so this stays causal.
FEATURE_COLS = ["energy_kwh", "temp_mean", "is_weekend", "is_holiday"]

HORIZON     = 7      # 1 week ahead
TEST_DAYS   = 30     # held out for rolling-origin evaluation
VAL_DAYS    = 60     # held out from train for early stopping

SEEDS       = (42, 7, 2024)   # ensemble of independent runs averaged at inference


@dataclass(frozen=True)
class Config:
    """Hyperparameters. Tweak only here — model is rebuilt from this object."""
    context_len: int = 56          # 8 weeks of history fed to the encoder
    patch_len:   int = 7           # one week per token
    stride:      int = 7           # non-overlapping patches → N=8 tokens
    n_channels:  int = 4           # energy + 3 exog
    d_model:     int = 96
    n_heads:     int = 4
    n_layers:    int = 3
    ff_mult:     int = 4           # FFN inner dim = d_model * ff_mult
    dropout:     float = 0.15

    lr:          float = 5e-4
    weight_decay: float = 1e-4
    batch_size:  int = 32
    max_epochs:  int = 200
    patience:    int = 25


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class PatchTST(nn.Module):
    """Multivariate PatchTST with RevIN on the target channel.

    Input  : (B, L, C)   energy + exogenous channels for the past L days
    Output : (B, H)      forecast of the target (channel 0) for H days

    Patching is channel-mixed: every patch is a (patch_len, C) block that gets
    flattened to a single token. This lets the transformer attend across
    channels at every patch position — appropriate when C is small (4) and
    the dataset is short (~1900 days).

    RevIN (Kim et al., ICLR 2022) is applied on the target channel only, so
    the head can stay in target units; un-RevIN restores the prediction scale.
    """

    def __init__(self, cfg: Config, horizon: int):
        super().__init__()
        self.cfg     = cfg
        self.horizon = horizon
        self.n_patches = (cfg.context_len - cfg.patch_len) // cfg.stride + 1

        self.patch_embed = nn.Linear(cfg.patch_len * cfg.n_channels, cfg.d_model)
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.n_patches, cfg.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = cfg.d_model,
            nhead           = cfg.n_heads,
            dim_feedforward = cfg.d_model * cfg.ff_mult,
            dropout         = cfg.dropout,
            batch_first     = True,
            activation      = "gelu",
            norm_first      = True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.head    = nn.Linear(self.n_patches * cfg.d_model, horizon)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C). RevIN on the target channel (index 0).
        target = x[..., 0]
        mu  = target.mean(dim=-1, keepdim=True)
        sig = target.std (dim=-1, keepdim=True) + 1e-5
        x_n = x.clone()
        x_n[..., 0] = (target - mu) / sig

        # Patchify channel-mixed: (B, N, P, C) → (B, N, P*C).
        patches = x_n.unfold(dimension=1, size=self.cfg.patch_len, step=self.cfg.stride)
        # After unfold the new patch dim is appended at the end, so shape is
        # (B, N, C, P). Move channels next to patch_len and flatten.
        patches = patches.permute(0, 1, 3, 2).contiguous()                  # (B, N, P, C)
        patches = patches.flatten(start_dim=-2)                             # (B, N, P*C)

        tokens = self.patch_embed(patches) + self.pos_embed                 # (B, N, D)
        tokens = self.dropout(tokens)
        z      = self.encoder(tokens)                                       # (B, N, D)
        flat   = z.flatten(start_dim=1)                                     # (B, N*D)
        y_n    = self.head(flat)                                            # (B, H)

        # Un-RevIN on the prediction (target units).
        return y_n * sig + mu


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def make_windows(features: np.ndarray, context_len: int, horizon: int, target_idx: int = 0):
    """Build sliding (context, target) pairs from a 2-D feature matrix.

    features: (T, C). Output X: (N, L, C), Y: (N, H) — target channel only.
    """
    T = len(features)
    n = T - context_len - horizon + 1
    if n <= 0:
        raise ValueError(f"Series too short: {T} < {context_len + horizon}")
    X = np.stack([features[i : i + context_len]                              for i in range(n)])
    Y = np.stack([features[i + context_len : i + context_len + horizon, target_idx] for i in range(n)])
    return X.astype(np.float32), Y.astype(np.float32)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train(cfg: Config, train_feats: np.ndarray, val_feats: np.ndarray,
          mean: np.ndarray, std: np.ndarray, device: torch.device):
    """Standard supervised training with cosine LR schedule + early stopping.

    `mean` / `std` are per-channel z-score stats computed on the train split
    only. RevIN inside the model normalises the *target* channel per-sample;
    the global z-score keeps every channel on a comparable numeric range
    before that.
    """
    train_n = (train_feats - mean) / std
    val_n   = (val_feats   - mean) / std
    X_tr, Y_tr = make_windows(train_n, cfg.context_len, HORIZON)
    X_va, Y_va = make_windows(val_n,   cfg.context_len, HORIZON)

    tr_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr)),
        batch_size=cfg.batch_size, shuffle=True, drop_last=False,
    )
    va_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(Y_va)),
        batch_size=cfg.batch_size,
    )

    model = PatchTST(cfg, HORIZON).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    loss_fn = nn.MSELoss()

    best_val = math.inf
    best_state = None
    bad = 0
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                va_losses.append(loss_fn(model(xb), yb).item())

        sched.step()
        tr_l = float(np.mean(tr_losses))
        va_l = float(np.mean(va_losses))
        history.append({"epoch": epoch, "train_loss": tr_l, "val_loss": va_l,
                        "lr": opt.param_groups[0]["lr"]})

        improved = va_l < best_val - 1e-6
        if improved:
            best_val = va_l
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % 10 == 0 or improved:
            tag = "  *" if improved else ""
            print(f"  epoch {epoch:3d}  train={tr_l:.4f}  val={va_l:.4f}{tag}")

        if bad >= cfg.patience:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_val, history


# --------------------------------------------------------------------------- #
# Rolling-origin inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rolling_predict(model: PatchTST, full_df: pd.DataFrame, test_start_idx: int,
                    mean: np.ndarray, std: np.ndarray, device: torch.device,
                    context_len: int, horizon: int, target_col: str):
    """Generate 7-day forecasts at each non-overlapping origin in the test set.

    At window i the history is `full_df[: test_start_idx + i*horizon]` and
    the model receives the last `context_len` standardised feature rows. The
    forecast is inverse-normalised back to kWh before being returned.
    """
    model.eval()
    feats   = full_df[FEATURE_COLS].values.astype(np.float32)
    actuals = full_df[target_col]
    target_mean, target_std = float(mean[0]), float(std[0])

    n_windows = (len(full_df) - test_start_idx) // horizon
    rows = []
    for i in range(n_windows):
        cutoff   = test_start_idx + i * horizon
        context  = (feats[cutoff - context_len:cutoff] - mean) / std        # (L, C)
        x        = torch.from_numpy(context[None, ...]).to(device)          # (1, L, C)
        y_hat    = model(x).cpu().numpy().ravel() * target_std + target_mean
        target   = actuals.iloc[cutoff : cutoff + horizon]
        for j, (date, actual) in enumerate(target.items()):
            rows.append({
                "window":     i + 1,
                "origin":     full_df.index[cutoff - 1],
                "date":       date,
                "step":       j + 1,
                "actual":     float(actual),
                "prediction": float(y_hat[j]),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    device = get_device()
    print(f"Device: {device}")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")
    feats = df[FEATURE_COLS].values.astype(np.float32)
    assert FEATURE_COLS[0] == TARGET_COL, "target must be channel 0"

    # Train / val / test split (test = last 30 days, val = 60 days before that)
    test_start = len(df) - TEST_DAYS
    val_start  = test_start - VAL_DAYS

    train_feats = feats[:val_start]
    val_feats   = feats[val_start - 200:test_start]   # 200-day warm-up so val
                                                       # can still build full windows
    mean = train_feats.mean(axis=0)
    std  = train_feats.std (axis=0) + 1e-8
    print(f"Train: {len(train_feats)}  Val: {len(val_feats)}  Test: {TEST_DAYS}")
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"  μ = {np.array2string(mean, precision=2)}")
    print(f"  σ = {np.array2string(std,  precision=2)}")

    cfg = Config()
    assert cfg.n_channels == len(FEATURE_COLS), "n_channels must match FEATURE_COLS"
    print(f"Config: {asdict(cfg)}")

    # Seed ensemble: train SEEDS independent runs and average their forecasts.
    # Reduces variance from initialisation noise — typical 5-10% MAE drop.
    seed_preds = []
    seed_val_mse = []
    for k, seed in enumerate(SEEDS, 1):
        print(f"\n=== Seed {k}/{len(SEEDS)} = {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        model, best_val, history = train(cfg, train_feats, val_feats, mean, std, device)
        print(f"  done in {time.time()-t0:.1f}s. best val MSE={best_val:.4f}")
        seed_val_mse.append(best_val)

        preds = rolling_predict(
            model, df, test_start_idx=test_start,
            mean=mean, std=std, device=device,
            context_len=cfg.context_len, horizon=HORIZON,
            target_col=TARGET_COL,
        )
        preds["seed"] = seed
        seed_preds.append(preds)

        torch.save(model.state_dict(), OUT_DIR / f"model_seed{seed}.pt")
        pd.DataFrame(history).to_csv(OUT_DIR / f"history_seed{seed}.csv", index=False)

    # Average predictions across seeds (group by date) — one row per day.
    all_preds = pd.concat(seed_preds, ignore_index=True)
    all_preds.to_csv(OUT_DIR / "predictions_per_seed.csv", index=False)

    ensemble = (
        all_preds.groupby(["window", "origin", "date", "step"], as_index=False)
                 .agg({"actual": "first", "prediction": "mean"})
                 .sort_values(["window", "step"])
                 .reset_index(drop=True)
    )
    ensemble.to_csv(OUT_DIR / "predictions.csv", index=False)

    with open(OUT_DIR / "best_config.json", "w") as f:
        json.dump({
            "config":       asdict(cfg),
            "horizon":      HORIZON,
            "mean":         mean.tolist(),
            "std":          std.tolist(),
            "val_mse_per_seed": seed_val_mse,
            "seeds":        list(SEEDS),
            "target_col":   TARGET_COL,
            "feature_cols": FEATURE_COLS,
            "test_days":    TEST_DAYS,
            "val_days":     VAL_DAYS,
        }, f, indent=2)

    # Quick sanity check — per-window MAE for the ensemble.
    summary = ensemble.groupby("window").apply(
        lambda g: pd.Series({
            "MAE":  float(np.mean(np.abs(g["actual"] - g["prediction"]))),
            "RMSE": float(np.sqrt(np.mean((g["actual"] - g["prediction"])**2))),
        }),
        include_groups=False,
    )
    print("\nPer-window quick metrics (kWh) — seed-ensemble:")
    print(summary.to_string())
    print(f"\nOverall mean MAE: {summary['MAE'].mean():.3f}  std: {summary['MAE'].std():.3f}")
    print(f"Artefacts saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
