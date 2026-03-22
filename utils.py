"""
utils.py — Shared helper functions for the FEMA PA Prediction project.
Import in any notebook with:  import sys; sys.path.append('../'); from utils import *
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Transforms ────────────────────────────────────────────────────────────────
def log_transform(series):
    """Apply log1p to a Series or array (safe for zeros)."""
    return np.log1p(series)

def inverse_log_transform(series):
    """Reverse log1p back to original dollar scale."""
    return np.expm1(series)

# ── Train/test split ──────────────────────────────────────────────────────────
def time_based_split(df, date_col, split_year):
    """
    Time-aware split: train on years < split_year, test on >= split_year.
    Always use this instead of random split for time-series data.
    """
    dates = pd.to_datetime(df[date_col], errors='coerce')
    train = df[dates.dt.year <  split_year].copy()
    test  = df[dates.dt.year >= split_year].copy()
    return train, test

# ── Feature engineering ───────────────────────────────────────────────────────
def get_season(month):
    """
    Map a month number (int) to a season string.
    Works with .apply(get_season) on a month Series.
    """
    if month in (12, 1, 2):  return "Winter"
    if month in (3,  4, 5):  return "Spring"
    if month in (6,  7, 8):  return "Summer"
    return "Fall"

def add_prior_disasters(df, state_col, date_col, window_years=5):
    """
    For each row in df, count prior disasters in the same state
    within the last `window_years` years.
    Efficient: computes at disaster level then merges back.

    df must have columns: [state_col, date_col, 'disasterNumber']
    Returns the same df with a new column 'prior_disasters_5yr'.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col)

    prior = []
    dates  = df[date_col].values
    states = df[state_col].values

    for i in range(len(df)):
        cutoff = dates[i] - np.timedelta64(window_years * 365, 'D')
        mask = (states[:i] == states[i]) & (dates[:i] >= cutoff)
        prior.append(mask.sum())

    df['prior_disasters_5yr'] = prior
    return df

# ── Metrics ───────────────────────────────────────────────────────────────────
def regression_metrics(y_true, y_pred, label="Model"):
    """
    Print and return MAE, RMSE, R², MAPE for a regression model.
    All metrics computed on the log-transformed scale passed in.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    print(f"\n{'='*45}")
    print(f"  {label}")
    print(f"{'='*45}")
    print(f"  MAE  : {mae:>10.4f}")
    print(f"  RMSE : {rmse:>10.4f}")
    print(f"  R²   : {r2:>10.4f}")
    print(f"  MAPE : {mape:>9.2f}%")
    return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

# ── Quick summary ─────────────────────────────────────────────────────────────
def data_summary(df, name="DataFrame"):
    """Print shape, null counts, and a peek at dtypes."""
    print(f"\n{'='*55}")
    print(f"  {name}  |  {df.shape[0]:,} rows  x  {df.shape[1]} cols")
    print(f"{'='*55}")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        print("Columns with nulls:")
        for col, n in nulls.items():
            print(f"  {col:<35} {n:>8,}  ({100*n/len(df):.1f}%)")
    else:
        print("  No nulls  ✓")
    print()
