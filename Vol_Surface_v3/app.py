"""
Volatility Surface v3 (Wrapper)

Guided CLI that orchestrates existing modules from Vol_Surface (v2) and 13_F.
Focuses on interactive guidance, P&L-oriented tips, and reusing code.
"""

import sys
from pathlib import Path

# Ensure we can import sibling projects
ROOT = Path(__file__).resolve().parents[1]
VOL = ROOT / 'Vol_Surface'
THIRTEEN_F = ROOT / '13_F'
sys.path.insert(0, str(VOL))
sys.path.insert(0, str(THIRTEEN_F))

# Ensure consistent UTF-8 console output without special arrows/smart quotes
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def pnl_tip(msg: str):
    print("\nUSAGE & P&L RATIONALE")
    print("-" * 80)
    print(msg)
    print("-" * 80)


def run_single_ticker():
    pnl_tip(
        "Single Ticker Analysis: fetch options, compute HV & seasonality, screen signals\n"
        "(calendars/diagonals), and rank strategies. P&L: buy vol when cheap seasonally\n"
        "and sell when rich with term carry, reducing false positives. Then save report."
    )
    from vol_analysis_system import VolatilityAnalysisSystem
    system = VolatilityAnalysisSystem()
    ticker = input("Enter ticker: ").strip().upper()
    data_source = 'yfinance'
    print("Analyzing (this may take ~15-30s)...")
    res = system.analyze_ticker(ticker, data_source=data_source)
    if isinstance(res, dict) and 'error' in res:
        print(f"Error: {res['error']}")
        return
    print("Analysis summary:")
    try:
        print(f"  Price: ${res['current_price']:.2f}")
        print(f"  HV(30d): {res['historical_vol']*100:.2f}%")
        print(f"  Signals: {len(res.get('alpha_signals', []))}")
        print(f"  Strategies ranked: {len(res.get('best_strategies', []))}")
    except Exception:
        pass
    gen = input("Generate HTML report now? (y/n): ").strip().lower()
    if gen == 'y':
        report = system.generate_report(ticker)
        print(f"Report: {report}")


def run_commodity():
    pnl_tip(
        "Commodity Analytics: seasonality + term structure + roll yield for ETFs\n"
        "(GLD/USO/UNG/WEAT/SOYB/PPLT/PALL). P&L: avoid selling artificially\n"
        "expensive IV in harvest/roll windows; capture roll yield with calendars/diagonals."
    )
    from vol_analysis_system import VolatilityAnalysisSystem
    system = VolatilityAnalysisSystem()
    ticker = input("Commodity ETF (e.g., GLD, USO, UNG, WEAT, SOYB, PPLT, PALL): ").strip().upper()
    try:
        path = system.generate_commodity_report(ticker)
        print(f"Commodity report: {path}")
    except Exception as e:
        print(f"Error: {e}")


def run_svi():
    pnl_tip(
        "SVI Surface Fitting: fit arbitrage-aware SVI per expiry; view diagnostics\n"
        "to spot smile/term mispricings. P&L: cleaner Greeks/prices -> better hedges;\n"
        "identify risk-reversals, butterflies, and calendars."
    )
    from svi_surface_fitter import SVISurfaceFitter
    fitter = SVISurfaceFitter()
    ticker = input("Enter ticker: ").strip().upper()
    try:
        out = fitter.fit_ticker(ticker, data_source='yfinance')
        print(f"SVI parameters: {out}")
    except Exception as e:
        print(f"Error: {e}")


def run_mc():
    pnl_tip(
        "Monte Carlo: BS (constant vol), Merton (jumps), Heston (stochastic vol).\n"
        "P&L: size positions to distribution tails; choose models per regime to\n"
        "avoid left-tail losses."
    )
    from vol_analysis_system import VolatilityAnalysisSystem
    from enhanced_monte_carlo import MonteCarloSimulator
    ticker = input("Enter ticker: ").strip().upper()
    print("Model: 1=BS, 2=Merton, 3=Heston")
    m = input("Choice: ").strip()
    model = {'1': 'BS', '2': 'MERTON', '3': 'HESTON'}.get(m, 'BS')
    days = int(input("Horizon (days, default 30): ") or '30')
    paths = int(input("Paths (default 5000): ") or '5000')
    params = {}
    if model == 'MERTON':
        params['lambda'] = float(input("lambda (0.2): ") or '0.2')
        params['muJ'] = float(input("muJ (-0.05): ") or '-0.05')
        params['sigmaJ'] = float(input("sigmaJ (0.10): ") or '0.10')
    if model == 'HESTON':
        params['v0'] = float(input("v0 (hv^2 default blank): ") or '0')
        params['kappa'] = float(input("kappa (2.0): ") or '2.0')
        params['theta'] = float(input("theta (hv^2 default blank): ") or '0')
        params['xi'] = float(input("xi (0.5): ") or '0.5')
        params['rho'] = float(input("rho (-0.5): ") or '-0.5')
    sim = MonteCarloSimulator()
    res = sim.run_ticker(
        ticker,
        model=model,
        horizon_days=days,
        paths=paths,
        data_source='yfinance',
        params=params,
    )
    # Build report using v2 helper
    system = VolatilityAnalysisSystem()
    report = system.generate_mc_report(ticker, model, res.get('summary'), res.get('paths_csv_path'))
    print(f"Monte Carlo report: {report}")


def run_13f_suggest():
    pnl_tip(
        "13F Smart-Money Discovery: surface popular holdings and consensus adds; optionally\n"
        "analyze a name with the options engine. P&L: concentrate capital in high-\n"
        "conviction stocks backed by top funds, then choose an options structure\n"
        "(calendars/diagonals) per term/seasonality."
    )
    import glob
    import pandas as pd
    runs_dir = THIRTEEN_F / 'runs'
    csvs = sorted(glob.glob(str(runs_dir / 'popular_holdings_*.csv')))
    if not csvs:
        print("No popular_holdings CSV found. Attempting a quick parse (limit=3)...")
        try:
            from parser import run_13f_parser
            run_13f_parser(limit=3)
            csvs = sorted(glob.glob(str(runs_dir / 'popular_holdings_*.csv')))
        except Exception as e:
            print(f"Could not generate runs: {e}")
            return
    if not csvs:
        print("Still no runs available.")
        return
    latest = csvs[-1]
    print(f"Using: {latest}")
    try:
        df = pd.read_csv(latest)
        print(df.head(20).to_string(index=False))
    except Exception as e:
        print(f"Error reading {latest}: {e}")
        return
    go = input("Analyze a ticker from this list with the options engine? (enter symbol or blank): ").strip().upper()
    if go:
        # Reuse single ticker flow
        from vol_analysis_system import VolatilityAnalysisSystem
        system = VolatilityAnalysisSystem()
        res = system.analyze_ticker(go, data_source='yfinance')
        if isinstance(res, dict) and 'error' in res:
            print(f"Error: {res['error']}")
        else:
            report = system.generate_report(go)
            print(f"Report: {report}")


def main():
    print("=" * 80)
    print("VOLATILITY SURFACE v3 - INTEGRATED WRAPPER")
    print("=" * 80)
    # Model explanations (static, shown at launch)
    print("\nModels Overview (for simulation & pricing):")
    print("- BS (Black-Scholes): constant volatility; fast, baseline for equities/ETFs.")
    print("- Merton (Jump-Diffusion): adds Poisson jumps; handles gap risk and fat tails.")
    print("- Heston (Stochastic Vol): mean-reverting variance with vol-of-vol and rho;")
    print("  calibrates to smiles/term. QuantLib calibration used if available.")
    print("\nGuided Paths (quick orientation):")
    print("  1) Single Ticker Deep Dive - Analyze + Report [vol_analysis_system.py]")
    print("  2) Commodity Analytics - Seasonality & Term [vol_analysis_system.py]")
    print("  3) Fit SVI Surface - Diagnostics [svi_surface_fitter.py]")
    print("  4) Monte Carlo (BS/Merton/Heston) - Report [enhanced_monte_carlo.py, vol_analysis_system.py]")
    print("  5) 13F Smart-Money Discovery -> Analyze [13_F/parser.py, vol_analysis_system.py]")
    print("  6) Exit")

    while True:
        print("\nMAIN MENU v3 (use number keys)")
        print(" 1. Single Ticker Deep Dive [vol_analysis_system.py]")
        print(" 2. Commodity Analytics [vol_analysis_system.py]")
        print(" 3. Fit SVI Surface [svi_surface_fitter.py]")
        print(" 4. Monte Carlo Simulation [enhanced_monte_carlo.py, vol_analysis_system.py]")
        print(" 5. 13F Smart-Money Discovery [13_F/parser.py, vol_analysis_system.py]")
        print(" 6. Exit")
        choice = input("Select (1-6): ").strip()
        if choice == '1':
            run_single_ticker()
        elif choice == '2':
            run_commodity()
        elif choice == '3':
            run_svi()
        elif choice == '4':
            run_mc()
        elif choice == '5':
            run_13f_suggest()
        elif choice == '6':
            print("Exiting v3...")
            break
        else:
            print("Invalid choice.")


if __name__ == '__main__':
    main()
