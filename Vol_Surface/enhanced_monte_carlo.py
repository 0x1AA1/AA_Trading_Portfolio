"""
Enhanced Monte Carlo Simulator

Implements Black-Scholes, Merton Jump-Diffusion, and Heston models for
underlying price path simulation and simple option P&L analysis.

Outputs are saved with timestamps under data_cache/vol_surface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np


MODEL_EXPLANATIONS = {
    "BS": "Black–Scholes: lognormal diffusion, constant volatility, no jumps.",
    "MERTON": "Merton Jump–Diffusion: BS + Poisson jumps with lognormal jump sizes.",
    "HESTON": "Heston: stochastic variance mean-reverting CIR process with correlation.",
}


@dataclass
class MCConfig:
    S0: float
    T: float  # years
    r: float = 0.0
    q: float = 0.0
    steps: int = 252
    paths: int = 5000
    seed: Optional[int] = 42


class MonteCarloSimulator:
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "data_cache" / "vol_surface"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _rng(seed: Optional[int]) -> np.random.Generator:
        return np.random.default_rng(seed)

    def simulate_bs(self, cfg: MCConfig, sigma: float) -> np.ndarray:
        dt = cfg.T / cfg.steps
        rng = self._rng(cfg.seed)
        Z = rng.standard_normal((cfg.paths, cfg.steps))
        drift = (cfg.r - cfg.q - 0.5 * sigma * sigma) * dt
        diff = sigma * np.sqrt(dt) * Z
        log_S = np.log(cfg.S0) + np.cumsum(drift + diff, axis=1)
        S = np.exp(log_S)
        S = np.concatenate([np.full((cfg.paths, 1), cfg.S0), S], axis=1)
        return S

    def simulate_merton(self, cfg: MCConfig, sigma: float, lam: float, muJ: float, sigmaJ: float) -> np.ndarray:
        dt = cfg.T / cfg.steps
        rng = self._rng(cfg.seed)
        Z = rng.standard_normal((cfg.paths, cfg.steps))
        # Jump component
        N = rng.poisson(lam * dt, size=(cfg.paths, cfg.steps))
        J = rng.normal(muJ, sigmaJ, size=(cfg.paths, cfg.steps))
        jump_term = (np.exp(J) - 1.0) * N

        # Compensate drift for jumps
        k = np.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1.0
        drift = (cfg.r - cfg.q - 0.5 * sigma * sigma - lam * k) * dt
        diff = sigma * np.sqrt(dt) * Z

        log_S = np.log(cfg.S0) + np.cumsum(drift + diff + np.log1p(jump_term + 1e-16), axis=1)
        S = np.exp(log_S)
        S = np.concatenate([np.full((cfg.paths, 1), cfg.S0), S], axis=1)
        return S

    def simulate_heston(
        self,
        cfg: MCConfig,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
    ) -> np.ndarray:
        dt = cfg.T / cfg.steps
        rng = self._rng(cfg.seed)
        Z1 = rng.standard_normal((cfg.paths, cfg.steps))
        Z2 = rng.standard_normal((cfg.paths, cfg.steps))
        Z2 = rho * Z1 + np.sqrt(max(1e-12, 1 - rho * rho)) * Z2

        v = np.full((cfg.paths,), max(1e-10, v0))
        S = np.full((cfg.paths,), cfg.S0)
        out = np.empty((cfg.paths, cfg.steps + 1))
        out[:, 0] = S

        for t in range(1, cfg.steps + 1):
            # CIR variance with full truncation Euler
            v = np.maximum(0.0, v + kappa * (theta - v) * dt + xi * np.sqrt(np.maximum(v, 0)) * np.sqrt(dt) * Z2[:, t - 1])
            S = S * np.exp((cfg.r - cfg.q - 0.5 * v) * dt + np.sqrt(np.maximum(v, 0)) * np.sqrt(dt) * Z1[:, t - 1])
            out[:, t] = S

        return out

    def summarize_terminal(self, paths: np.ndarray) -> Dict[str, float]:
        terminal = paths[:, -1]
        ret = terminal / paths[:, 0] - 1.0
        return {
            "S0": float(np.mean(paths[:, 0])),
            "S_T_mean": float(np.mean(terminal)),
            "S_T_std": float(np.std(terminal)),
            "ret_mean": float(np.mean(ret)),
            "ret_std": float(np.std(ret)),
            "p_up": float(np.mean(terminal > paths[:, 0])),
            "p_10pct_drop": float(np.mean(terminal < 0.9 * paths[:, 0])),
        }

    def run_ticker(
        self,
        ticker: str,
        model: str = "BS",
        horizon_days: int = 30,
        paths: int = 5000,
        data_source: str = "yfinance",
        params: Optional[Dict] = None,
    ) -> Dict:
        from data_manager import OptionsDataManager

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dm = OptionsDataManager(cache_dir=self.cache_dir)
        S0 = dm.get_current_price(ticker, data_source=data_source)
        if S0 <= 0:
            raise ValueError("Unable to obtain current price for simulation")

        T = max(1, horizon_days) / 365.0
        cfg = MCConfig(S0=S0, T=T, steps=min(252, max(30, horizon_days)), paths=paths)

        # Default parameters
        hv = max(dm.calculate_historical_volatility(ticker, window=30, data_source=data_source), 1e-4)
        res_paths: np.ndarray

        if model.upper() == "BS":
            res_paths = self.simulate_bs(cfg, sigma=hv)
        elif model.upper() == "MERTON":
            p = params or {}
            lam = float(p.get("lambda", 0.2))
            muJ = float(p.get("muJ", -0.05))
            sigmaJ = float(p.get("sigmaJ", 0.10))
            res_paths = self.simulate_merton(cfg, sigma=hv, lam=lam, muJ=muJ, sigmaJ=sigmaJ)
        elif model.upper() == "HESTON":
            p = params or {}
            v0 = float(p.get("v0", hv * hv))
            kappa = float(p.get("kappa", 2.0))
            theta = float(p.get("theta", hv * hv))
            xi = float(p.get("xi", 0.5))
            rho = float(p.get("rho", -0.5))
            res_paths = self.simulate_heston(cfg, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
        else:
            raise ValueError("Unsupported model")

        summary = self.summarize_terminal(res_paths)
        summary.update({
            "ticker": ticker,
            "model": model.upper(),
            "explanation": MODEL_EXPLANATIONS.get(model.upper(), ""),
            "timestamp": ts,
            "paths": cfg.paths,
            "steps": cfg.steps,
        })

        # Save outputs
        ticker_dir = self.cache_dir / ticker.upper()
        ticker_dir.mkdir(parents=True, exist_ok=True)
        json_path = ticker_dir / f"{ticker}_mc_{model.lower()}_{ts}.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Optionally save sample paths CSV (downsampled)
        sample_idx = slice(0, min(50, res_paths.shape[0]))
        csv_path = ticker_dir / f"{ticker}_mc_{model.lower()}_paths_{ts}.csv"
        np.savetxt(csv_path, res_paths[sample_idx, :], delimiter=",", fmt="%.6f")

        return {
            "summary_path": str(json_path),
            "paths_csv_path": str(csv_path),
            "summary": summary,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Monte Carlo")
    parser.add_argument("ticker")
    parser.add_argument("--model", default="BS", choices=["BS", "MERTON", "HESTON"]) 
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--paths", type=int, default=5000)
    args = parser.parse_args()

    sim = MonteCarloSimulator()
    out = sim.run_ticker(args.ticker, model=args.model, horizon_days=args.days, paths=args.paths)
    print(json.dumps(out["summary"], indent=2))
