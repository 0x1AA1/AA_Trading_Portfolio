"""
Seasonality Features

Computes historical seasonality baselines for realized volatility (HV) and
evaluates current ATM IV per expiration against seasonality (z-scores).

Outputs are cached under per-ticker folder: {TICKER}/features/.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict

import numpy as np
import pandas as pd


def compute_hv_seasonality_baseline(
    ticker: str,
    cache_dir: Path,
    years: int = 3
) -> Path:
    """
    Build month-of-year baseline for 30d realized volatility over the past N years.
    Saves CSV to {TICKER}/features/hv_seasonality.csv
    """
    try:
        import yfinance as yf
    except Exception:
        raise RuntimeError("yfinance required for seasonality baselines")

    ticker_dir = cache_dir / ticker.upper() / 'features'
    ticker_dir.mkdir(parents=True, exist_ok=True)
    out_csv = ticker_dir / 'hv_seasonality.csv'

    end = datetime.now()
    start = end - timedelta(days=365 * max(1, years))
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start, end=end)
    if hist.empty:
        # Write empty
        pd.DataFrame(columns=['month', 'hv30_median', 'hv30_std']).to_csv(out_csv, index=False)
        return out_csv

    # Daily log returns
    ret = np.log(hist['Close'] / hist['Close'].shift(1))
    hv30 = ret.rolling(30).std() * np.sqrt(252)
    df = pd.DataFrame({'date': ret.index, 'hv30': hv30.values})
    df = df.dropna()
    df['month'] = df['date'].dt.month

    base = df.groupby('month')['hv30'].agg(['median', 'std']).reset_index()
    base = base.rename(columns={'median': 'hv30_median', 'std': 'hv30_std'})
    base.to_csv(out_csv, index=False)
    return out_csv


def evaluate_iv_vs_hv_seasonality(
    options_df: pd.DataFrame,
    current_price: float,
    hv_seasonality_csv: Path
) -> pd.DataFrame:
    """
    For each expiration, compute ATM IV and compare against HV seasonality baseline
    for the month of expiration. Returns a DataFrame with z-score per expiry.
    """
    if options_df.empty or 'impliedVolatility' not in options_df.columns:
        return pd.DataFrame()

    try:
        base = pd.read_csv(hv_seasonality_csv)
    except Exception:
        return pd.DataFrame()

    if base.empty or 'month' not in base.columns:
        return pd.DataFrame()

    df = options_df.copy()
    df['expiration'] = pd.to_datetime(df['expiration'])
    df['month'] = df['expiration'].dt.month
    # ATM selection per expiry
    df['strike_diff'] = (df['strike'] - current_price).abs()
    atm = df.sort_values(['expiration', 'strike_diff']).groupby('expiration').head(2)
    atm_iv = atm.groupby(['expiration', 'month'])['impliedVolatility'].mean().reset_index()

    merged = pd.merge(atm_iv, base, how='left', on='month')
    merged['z_seasonal'] = (merged['impliedVolatility'] - merged['hv30_median']) / merged['hv30_std'].replace(0, np.nan)
    merged['z_seasonal'] = merged['z_seasonal'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    merged = merged.rename(columns={'impliedVolatility': 'atm_iv'})
    merged = merged[['expiration', 'month', 'atm_iv', 'hv30_median', 'hv30_std', 'z_seasonal']]
    return merged

