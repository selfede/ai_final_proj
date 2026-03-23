"""
utils.py — Shared helper functions for the FEMA PA Prediction project.
Import in any notebook with:  import sys; sys.path.append('../'); from utils import *
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Classification bracket constants ─────────────────────────────────────────
# Project-level tiers (per-grant amount)
PROJECT_BINS   = [0, 10_000, 131_100, 1_000_000, float('inf')]
PROJECT_LABELS = {0: 'Micro (<$10k)', 1: 'Small ($10k–$131k)',
                  2: 'Large ($131k–$1M)', 3: 'Major (>$1M)'}

# Disaster-level tiers (total payout per disaster event)
DISASTER_BINS   = [0, 1_000_000, 50_000_000, 500_000_000, float('inf')]
DISASTER_LABELS = {0: 'Minor (<$1M)', 1: 'Moderate ($1M–$50M)',
                   2: 'Major ($50M–$500M)', 3: 'Catastrophic (>$500M)'}

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
from sklearn.metrics import accuracy_score, f1_score, classification_report

def classification_metrics(y_true, y_pred, label="Model", target_names=None):
    """
    Print and return Accuracy and Weighted F1 for a classification model.
    Also prints the full sklearn classification report.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n{'='*45}")
    print(f"  {label}")
    print(f"{'='*45}")
    print(f"  Accuracy   : {acc:>8.4f}")
    print(f"  F1 (wtd)   : {f1:>8.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=target_names,
                                zero_division=0))
    return {"label": label, "Accuracy": acc, "F1_weighted": f1}

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
