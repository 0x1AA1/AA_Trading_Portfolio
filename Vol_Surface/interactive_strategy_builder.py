"""
Interactive Option Strategy Builder

Allows users to:
1. Build custom option strategies leg by leg
2. Analyze payoffs and risk metrics
3. Run Monte Carlo simulations
4. Compare multiple strategies
"""

from payoff_scenario_analysis import PayoffAnalyzer, create_common_strategy
import numpy as np
from typing import Optional

try:
    # Optional: use data manager to source spot, IV, ATM premiums
    from data_manager import OptionsDataManager
except Exception:
    OptionsDataManager = None  # type: ignore


def interactive_strategy_builder(ticker: Optional[str] = None, data_source: str = 'yfinance'):
    """Interactive command-line strategy builder

    If ticker is provided, fetches current price, historical volatility, and
    attempts to suggest ATM call/put premiums from the nearest expiration.
    """

    print("=" * 80)
    print("INTERACTIVE OPTION STRATEGY BUILDER")
    print("=" * 80)

    # Determine current price and context
    suggested_call = None
    suggested_put = None
    hv = None

    if ticker and OptionsDataManager is not None:
        dm = OptionsDataManager()
        try:
            current_price = float(dm.get_current_price(ticker, data_source=data_source))
        except Exception:
            current_price = float(input("\nEnter current stock price: $"))

        try:
            hv = float(dm.calculate_historical_volatility(ticker, window=30, data_source=data_source))
        except Exception:
            hv = None

        # Try to get ATM premiums from nearest expiration (>= 21 days)
        try:
            df = dm.fetch_options_data(ticker, data_source=data_source)
            if not df.empty and {'strike','option_type','expiration','mid'}.issubset(df.columns):
                dfc = df.copy()
                dfc['dte'] = (dfc['expiration'] - np.datetime64('today')) / np.timedelta64(1,'D')
                dfc = dfc[dfc['dte'] >= 21]
                if not dfc.empty:
                    # choose nearest expiration to ~30 days
                    dfc['dte_diff'] = (dfc['dte'] - 30).abs()
                    exp = dfc.sort_values('dte_diff').iloc[0]['expiration']
                    g = dfc[dfc['expiration'] == exp].copy()
                    g['strike_diff'] = (g['strike'] - current_price).abs()
                    atm_strike = g.sort_values('strike_diff').iloc[0]['strike']
                    call = g[(g['option_type']=='call') & (g['strike']==atm_strike)]
                    put = g[(g['option_type']=='put') & (g['strike']==atm_strike)]
                    if not call.empty:
                        suggested_call = float(call.iloc[0].get('mid') or call.iloc[0].get('lastPrice') or 0.0)
                    if not put.empty:
                        suggested_put = float(put.iloc[0].get('mid') or put.iloc[0].get('lastPrice') or 0.0)
        except Exception:
            pass
    else:
        # Fallback: prompt user for price
        current_price = float(input("\nEnter current stock price: $"))

    # Display context if available
    if ticker:
        print(f"\nContext for {ticker} (source: {data_source}):")
        print(f"  Spot: ${current_price:.2f}")
        if hv is not None:
            print(f"  Historical Vol (30d): {hv*100:.2f}%")
        if suggested_call is not None or suggested_put is not None:
            print("  Suggested ATM premiums (nearest ~30d):")
            if suggested_call is not None:
                print(f"    Call ATM ~ ${suggested_call:.2f}")
            if suggested_put is not None:
                print(f"    Put  ATM ~ ${suggested_put:.2f}")

    # Choose mode
    print("\nSelect mode:")
    print("1. Build custom strategy (add legs manually)")
    print("2. Use common strategy template")

    mode = input("Choice (1 or 2): ").strip()

    analyzer = PayoffAnalyzer(current_price)

    if mode == '1':
        # Custom strategy
        print("\n" + "=" * 80)
        print("BUILD CUSTOM STRATEGY")
        print("=" * 80)

        while True:
            print("\nAdd a leg:")
            option_type = input("  Option type (call/put) or 'done' to finish: ").strip().lower()

            if option_type == 'done':
                break

            if option_type not in ['call', 'put']:
                print("  Invalid option type. Use 'call' or 'put'")
                continue

            strike = float(input("  Strike price: $"))
            premium = float(input("  Premium: $"))
            quantity = int(input("  Quantity (default 1): ") or "1")
            position = input("  Position (long/short, default long): ").strip().lower() or "long"

            if position not in ['long', 'short']:
                print("  Invalid position. Use 'long' or 'short'")
                continue

            analyzer.add_leg(option_type, strike, premium, quantity, position)
            print(f"  Added: {position.upper()} {quantity}x {option_type.upper()} ${strike} @ ${premium}")

    else:
        # Template strategy
        print("\n" + "=" * 80)
        print("COMMON STRATEGY TEMPLATES")
        print("=" * 80)
        print("\nAvailable strategies:")
        print("  1. Long Call")
        print("  2. Long Put")
        print("  3. Long Straddle")
        print("  4. Long Strangle")
        print("  5. Bull Call Spread")
        print("  6. Bear Put Spread")
        print("  7. Iron Condor")
        print("  8. Butterfly")
        print("  9. Short Straddle")
        print(" 10. Short Strangle")

        choice = input("\nSelect strategy (1-10): ").strip()

        strategy_map = {
            '1': 'long_call',
            '2': 'long_put',
            '3': 'long_straddle',
            '4': 'long_strangle',
            '5': 'bull_call_spread',
            '6': 'bear_put_spread',
            '7': 'iron_condor',
            '8': 'butterfly',
            '9': 'short_straddle',
            '10': 'short_strangle'
        }

        if choice not in strategy_map:
            print("Invalid choice")
            return

        strategy_name = strategy_map[choice]

        # Get premium estimates
        atm_premium = current_price * 0.03  # Default 3%
        if suggested_call is not None and suggested_put is not None:
            use_custom = input(f"\nUse suggested ATM premiums (Call ${suggested_call:.2f} / Put ${suggested_put:.2f})? (y/n): ").strip().lower()
        else:
            use_custom = input(f"\nUse default ATM premium (${atm_premium:.2f})? (y/n): ").strip().lower()

        if use_custom == 'n':
            atm_premium_call = float(input("Enter ATM call premium: $"))
            atm_premium_put = float(input("Enter ATM put premium: $"))
        else:
            if suggested_call is not None and suggested_put is not None:
                atm_premium_call = suggested_call
                atm_premium_put = suggested_put
            else:
                atm_premium_call = atm_premium
                atm_premium_put = atm_premium

        analyzer = create_common_strategy(
            strategy_name, current_price,
            atm_premium_call, atm_premium_put
        )

    # Analysis
    if len(analyzer.legs) == 0:
        print("\nNo legs added. Exiting.")
        return

    print("\n" + "=" * 80)
    print("STRATEGY ANALYSIS")
    print("=" * 80)

    analyzer.print_summary()

    # Payoff diagram
    print("\nGenerating payoff diagram...")
    fig_payoff = analyzer.plot_payoff(
        title=f"Payoff Diagram - {len(analyzer.legs)} Legs",
        save_html=f"strategy_payoff_{int(current_price)}.html"
    )
    print(f"Saved: strategy_payoff_{int(current_price)}.html")

    # Monte Carlo simulation
    run_mc = input("\nRun Monte Carlo simulation? (y/n): ").strip().lower()

    if run_mc == 'y':
        time_to_expiry = float(input("Time to expiration (years, e.g., 0.25 for 3 months): "))
        volatility = float(input("Expected volatility (e.g., 0.30 for 30%): "))
        n_sims = int(input("Number of simulations (default 10000): ") or "10000")

        print(f"\nRunning {n_sims} simulations...")
        mc_results = analyzer.monte_carlo_simulation(time_to_expiry, volatility, n_sims)

        print("\n" + "=" * 80)
        print("MONTE CARLO RESULTS")
        print("=" * 80)
        print(f"Probability of Profit: {mc_results['prob_profit']:.2%}")
        print(f"Expected Value: ${mc_results['expected_value']:.2f}")
        print(f"Average Profit (when profitable): ${mc_results['avg_profit']:.2f}")
        print(f"Average Loss (when losing): ${mc_results['avg_loss']:.2f}")
        print(f"Max Profit: ${mc_results['max_profit']:.2f}")
        print(f"Max Loss: ${mc_results['max_loss']:.2f}")
        print(f"Value at Risk (95%): ${mc_results['var_95']:.2f}")
        print(f"Conditional VaR (95%): ${mc_results['cvar_95']:.2f}")
        print(f"Standard Deviation: ${mc_results['std_dev']:.2f}")

        # Plot MC results
        print("\nGenerating Monte Carlo distribution plots...")
        fig_mc = analyzer.plot_monte_carlo(
            mc_results,
            title=f"Monte Carlo Analysis - {n_sims} Simulations",
            save_html=f"strategy_monte_carlo_{int(current_price)}.html"
        )
        print(f"Saved: strategy_monte_carlo_{int(current_price)}.html")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("Open the HTML files in your browser to view interactive charts!")


if __name__ == "__main__":
    # Optional CLI: allow entering a ticker to auto-populate
    try:
        t = input("Enter ticker for auto-context (blank to skip): ").strip().upper()
    except Exception:
        t = ''
    interactive_strategy_builder(ticker=t or None)
