"""
SVI Surface Fitter

Fits Stochastic Volatility Inspired (SVI) parameterization to option smiles
for each expiration and saves calibrated parameters and diagnostics.

Outputs are written to data_cache/vol_surface with timestamps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


@dataclass
class SVIParams:
    expiration: np.datetime64
    T: float
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    rmse: float
    n: int


class SVISurfaceFitter:
    """
    Calibrate raw SVI (a, b, rho, m, sigma) per expiry.

    Model:
        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
        iv(k) = sqrt(w(k) / T)

    We approximate forward F with spot when rates/dividends are unknown.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "data_cache" / "vol_surface"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _svi_total_variance(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    @staticmethod
    def _bounded_params(x: np.ndarray) -> Tuple[float, float, float, float, float]:
        a, b, rho, m, sigma = x
        # Enforce simple constraints
        b = max(b, 1e-8)
        sigma = max(sigma, 1e-8)
        rho = float(np.clip(rho, -0.999, 0.999))
        return a, b, rho, m, sigma

    def _fit_expiry(self, k: np.ndarray, iv: np.ndarray, T: float) -> Tuple[SVIParams, np.ndarray]:
        try:
            from scipy.optimize import minimize
        except Exception:
            raise RuntimeError("SciPy required for SVI fitting. Please install scipy.")

        # Initial guess (a ~ atm var, b ~ slope, rho ~ 0, m ~ 0, sigma ~ 0.1)
        atm_var = float(np.nanmedian((iv ** 2) * T)) if len(iv) else 0.04
        x0 = np.array([atm_var, 0.1, 0.0, 0.0, 0.1], dtype=float)

        def loss(x: np.ndarray) -> float:
            a, b, rho, m, sigma = self._bounded_params(x)
            w = self._svi_total_variance(k, a, b, rho, m, sigma)
            # Guard against negative due to numerical issues
            w = np.maximum(w, 1e-10)
            model_iv = np.sqrt(w / max(T, 1e-8))
            err = model_iv - iv
            # L2 with gentle L2 regularization on parameters
            reg = 1e-4 * (a * a + b * b + m * m + sigma * sigma + rho * rho)
            return float(np.nanmean(err ** 2) + reg)

        res = minimize(loss, x0, method="L-BFGS-B")

        a, b, rho, m, sigma = self._bounded_params(res.x)
        w = self._svi_total_variance(k, a, b, rho, m, sigma)
        w = np.maximum(w, 1e-10)
        model_iv = np.sqrt(w / max(T, 1e-8))
        rmse = float(np.sqrt(np.nanmean((model_iv - iv) ** 2)))

        params = SVIParams(
            expiration=np.datetime64('1970-01-01'),  # placeholder, set by caller
            T=T,
            a=float(a),
            b=float(b),
            rho=float(rho),
            m=float(m),
            sigma=float(sigma),
            rmse=rmse,
            n=int(len(k)),
        )
        return params, model_iv

    def fit_ticker(self, ticker: str, data_source: str = "yfinance") -> Path:
        """
        Fit SVI per expiration for a ticker and save parameters to CSV.
        Returns path to the saved CSV file.
        """
        from data_manager import OptionsDataManager

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker_dir = self.cache_dir / ticker.upper()
        ticker_dir.mkdir(parents=True, exist_ok=True)
        out_csv = ticker_dir / f"{ticker}_svi_params_{ts}.csv"

        dm = OptionsDataManager(cache_dir=self.cache_dir)
        df = dm.fetch_options_data(ticker, data_source=data_source, use_cache=True, cache_minutes=60)
        if df.empty:
            # Create empty file to signal attempt
            out_csv.write_text("")
            return out_csv

        # Keep rows with IV and required fields
        cols_ok = {'strike', 'expiration', 'impliedVolatility'} <= set(df.columns)
        if not cols_ok:
            out_csv.write_text("")
            return out_csv

        # Current price as proxy for forward
        spot = dm.get_current_price(ticker, data_source=data_source)
        if spot <= 0:
            spot = float(df.get('underlyingPrice', pd.Series([np.nan])).dropna().median() if 'underlyingPrice' in df.columns else 0) or 1.0

        records: List[SVIParams] = []
        rows: List[str] = ["expiration,T,a,b,rho,m,sigma,rmse,n"]

        for exp, g in df.groupby('expiration'):
            g = g.copy()
            # Guard: ensure enough strikes with IV
            g = g[np.isfinite(g['impliedVolatility']) & (g['impliedVolatility'] > 0)]
            if len(g) < 8:
                continue

            T = max(((np.datetime64(exp) - np.datetime64('today')) / np.timedelta64(1, 'D')), 1.0) / 365.0
            strikes = g['strike'].values.astype(float)
            # Log-moneyness relative to spot (proxy for forward)
            k = np.log(np.maximum(strikes, 1e-6) / max(spot, 1e-6))
            iv = g['impliedVolatility'].values.astype(float)

            try:
                p, model_iv = self._fit_expiry(k, iv, T)
                p.expiration = np.datetime64(exp)
                records.append(p)
                rows.append(
                    f"{np.datetime_as_string(p.expiration, unit='D')},{p.T:.8f},{p.a:.8f},{p.b:.8f},{p.rho:.6f},{p.m:.8f},{p.sigma:.8f},{p.rmse:.6f},{p.n}"
                )
            except Exception:
                continue

        out_csv.write_text("\n".join(rows))
        # After fitting, generate a small diagnostics report if possible
        try:
            diag = self.generate_diagnostics_report(ticker, out_csv, data_source=data_source)
        except Exception:
            diag = None
        return out_csv

    def reconstruct_iv(self, k: np.ndarray, params: SVIParams) -> np.ndarray:
        w = self._svi_total_variance(k, params.a, params.b, params.rho, params.m, params.sigma)
        w = np.maximum(w, 1e-10)
        return np.sqrt(w / max(params.T, 1e-8))


    def generate_diagnostics_report(self, ticker: str, params_csv: Path, data_source: str = 'yfinance') -> Optional[Path]:
        """
        Create a lightweight diagnostics HTML comparing fitted SVI smiles to market IV
        for up to 3 expirations.
        """
        try:
            import pandas as pd
            import plotly.graph_objects as go
        except Exception:
            return None

        from data_manager import OptionsDataManager

        df_params = pd.read_csv(params_csv)
        if df_params.empty:
            return None

        dm = OptionsDataManager(cache_dir=self.cache_dir)
        options = dm.fetch_options_data(ticker, data_source=data_source, use_cache=True, cache_minutes=60)
        if options.empty or 'impliedVolatility' not in options.columns:
            return None

        # Choose up to 3 expirations with most strikes
        counts = options.groupby('expiration').size().sort_values(ascending=False)
        chosen_exps = list(counts.index[:3])

        fig = go.Figure()

        spot = dm.get_current_price(ticker, data_source=data_source)
        spot = spot if spot > 0 else float(options.get('underlyingPrice', pd.Series([np.nan])).dropna().median() or 1.0)

        for exp in chosen_exps:
            g = options[options['expiration'] == exp].copy()
            g = g[np.isfinite(g['impliedVolatility']) & (g['impliedVolatility'] > 0)]
            if g.empty:
                continue
            T = max(((pd.Timestamp(exp) - pd.Timestamp.now()).days) / 365.0, 1/365)
            # Market smile
            fig.add_trace(go.Scatter(x=g['strike'], y=g['impliedVolatility']*100, mode='markers', name=f"Market {pd.Timestamp(exp).date()}"))

            # Find fitted params row
            row = df_params[df_params['expiration'] == pd.Timestamp(exp).strftime('%Y-%m-%d')]
            if row.empty:
                continue
            p = SVIParams(
                expiration=np.datetime64(pd.Timestamp(exp).date()),
                T=float(row.iloc[0]['T']),
                a=float(row.iloc[0]['a']),
                b=float(row.iloc[0]['b']),
                rho=float(row.iloc[0]['rho']),
                m=float(row.iloc[0]['m']),
                sigma=float(row.iloc[0]['sigma']),
                rmse=float(row.iloc[0]['rmse']),
                n=int(row.iloc[0]['n'])
            )
            strikes = np.linspace(float(g['strike'].min()), float(g['strike'].max()), 100)
            k = np.log(np.maximum(strikes, 1e-6) / max(spot, 1e-6))
            fit_iv = self.reconstruct_iv(k, p)
            fig.add_trace(go.Scatter(x=strikes, y=fit_iv*100, mode='lines', name=f"SVI fit {pd.Timestamp(exp).date()} (RMSE {p.rmse:.3f})"))

        fig.update_layout(title=f"SVI Diagnostics - {ticker}", xaxis_title='Strike', yaxis_title='IV (%)')
        reports = self.cache_dir / ticker.upper() / 'reports'
        reports.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_html = reports / f"{ticker}_svi_diagnostics_{ts}.html"
        fig.write_html(out_html)
        return out_html


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="SVI fitter")
    parser.add_argument("ticker")
    parser.add_argument("--source", default="yfinance")
    args = parser.parse_args()

    fitter = SVISurfaceFitter()
    path = fitter.fit_ticker(args.ticker, data_source=args.source)
    print(f"Saved SVI parameters to {path}")
