Volatility Surface v3 — Guided Wrapper
======================================

Purpose
- Provide a concise, guided CLI on top of the existing Vol_Surface (v2) and 13_F systems.
- Help route users to the right analysis path with P&L‑oriented tips and clear outputs.

Quick Start
- From the repo root: `py Finance_Trading_Coding/Vol_Surface_v3/app.py`
- At launch, the app explains BS, Merton, and Heston models (static overview).

Menu (naming convention)
- Each option states its purpose and the primary file(s) it calls in brackets:
  1) Single Ticker Deep Dive – Analyze + Report [`vol_analysis_system.py`]
  2) Commodity Analytics – Seasonality & Term [`vol_analysis_system.py`]
  3) Fit SVI Surface – Diagnostics [`svi_surface_fitter.py`]
  4) Monte Carlo (BS/Merton/Heston) – Report [`enhanced_monte_carlo.py`, `vol_analysis_system.py`]
  5) 13F Smart‑Money Discovery → Analyze [`13_F/parser.py`, `vol_analysis_system.py`]
  6) Exit

Outputs
- Reports and artifacts are saved under `Finance_Trading_Coding/data_cache/vol_surface/{TICKER}/reports` with timestamps.
- 13F runs reside under `Finance_Trading_Coding/13_F/runs/` (e.g., `popular_holdings_*.csv`).

Data Sources
- Default: `yfinance` (no credentials). Prefer IBKR (`Algo_Trade_IBKR/ibkr_api`) when available.

Notes
- If QuantLib is installed, Heston calibration may be used by v2 modules; otherwise, graceful fallbacks apply.
- For commodity ETFs (GLD, USO, UNG, WEAT, SOYB, PPLT, PALL), the commodity report includes seasonality and term insights.

Troubleshooting
- If network is restricted, data fetches may fail. Generate reports from cached data where possible.
