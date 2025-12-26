"""
Integrated Volatility Analysis System

Unified system for:
1. Multi-ticker options data fetching (IBKR/yfinance)
2. Volatility surface construction and analysis
3. Alpha generation through IV vs HV comparison
4. Strategy optimization and selection
5. Sensitivity analysis and Greeks
6. Interactive reporting

Architecture:
    Data Layer -> Analysis Layer -> Strategy Layer -> Reporting Layer
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "Algo_Trade_IBKR" / "ibkr_api"))

from greeks_calculator import GreeksCalculator
from vol_arbitrage import VolatilityArbitrageStrategies
from svi_surface_fitter import SVISurfaceFitter
from enhanced_monte_carlo import MonteCarloSimulator, MODEL_EXPLANATIONS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VolatilityAnalysisSystem:
    """
    Main orchestrator for integrated volatility analysis
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the volatility analysis system

        Args:
            cache_dir: Directory for data caching
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "data_cache" / "vol_surface"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.greeks_calc = GreeksCalculator()
        self.vol_arb = VolatilityArbitrageStrategies(self.greeks_calc)

        self.results = {}

        logger.info(f"Volatility Analysis System initialized (cache: {self.cache_dir})")

    def analyze_ticker(
        self,
        ticker: str,
        data_source: str = 'yfinance',
        sec_type: str = 'STK',
        exchange: str = 'SMART',
        currency: str = 'USD',
        hv_window: int = 30
    ) -> Dict:
        """
        Comprehensive analysis for a single ticker

        Args:
            ticker: Symbol to analyze
            data_source: 'yfinance' or 'ibkr'
            sec_type: Security type for IBKR
            exchange: Exchange for IBKR
            currency: Currency for IBKR
            hv_window: Historical volatility window (days)

        Returns:
            Dictionary with complete analysis results
        """
        logger.info(f"Analyzing {ticker} using {data_source}...")

        # Import data manager
        from data_manager import OptionsDataManager

        data_mgr = OptionsDataManager(cache_dir=self.cache_dir)

        # Fetch data
        options_df = data_mgr.fetch_options_data(
            ticker=ticker,
            data_source=data_source,
            sec_type=sec_type,
            exchange=exchange,
            currency=currency
        )

        if options_df.empty:
            logger.error(f"No options data for {ticker}")
            return {'error': 'No data available'}

        # Get current price and historical volatility
        current_price = data_mgr.get_current_price(ticker, data_source)
        historical_vol = data_mgr.calculate_historical_volatility(ticker, window=hv_window)

        # Build seasonality features (HV baseline + ATM IV by expiry)
        try:
            from seasonality_features import compute_hv_seasonality_baseline, evaluate_iv_vs_hv_seasonality
            base_path = compute_hv_seasonality_baseline(ticker, self.cache_dir)
            seasonality_df = evaluate_iv_vs_hv_seasonality(options_df, current_price, base_path)
        except Exception:
            seasonality_df = None

        # Perform alpha screening
        from alpha_screener import AlphaScreener
        screener = AlphaScreener(self.greeks_calc, self.vol_arb)

        alpha_signals = screener.screen_opportunities(
            options_df=options_df,
            current_price=current_price,
            historical_vol=historical_vol,
            ticker=ticker,
            seasonality_df=seasonality_df
        )

        # Strategy optimization
        from strategy_optimizer import StrategyOptimizer
        optimizer = StrategyOptimizer(self.greeks_calc)

        best_strategies = optimizer.rank_strategies(
            options_df=options_df,
            current_price=current_price,
            historical_vol=historical_vol,
            alpha_signals=alpha_signals
        )

        # Compile results
        results = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'data_source': data_source,
            'current_price': current_price,
            'historical_vol': historical_vol,
            'n_options': len(options_df),
            'alpha_signals': alpha_signals,
            'best_strategies': best_strategies,
            'options_data': options_df,
            'seasonality': None if seasonality_df is None else seasonality_df.to_dict('records')
        }

        self.results[ticker] = results

        # Enhanced logging with detailed metrics
        logger.info(f"="*60)
        logger.info(f"ANALYSIS COMPLETE FOR {ticker}")
        logger.info(f"="*60)
        logger.info(f"Current Price: ${current_price:.2f}")
        logger.info(f"Historical Volatility (30d): {historical_vol*100:.2f}%")
        logger.info(f"Options Analyzed: {len(options_df)}")
        logger.info(f"-"*60)
        logger.info(f"ALPHA SIGNALS DETECTED: {len(alpha_signals)} total")

        # Breakdown by type
        signal_types = {}
        for sig in alpha_signals:
            sig_type = sig.get('type', 'UNKNOWN')
            signal_types[sig_type] = signal_types.get(sig_type, 0) + 1

        for sig_type, count in sorted(signal_types.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  - {sig_type}: {count} signals")

        # Top signals summary
        high_conf_signals = [s for s in alpha_signals if s.get('confidence') == 'HIGH']
        logger.info(f"  - HIGH confidence: {len(high_conf_signals)}")
        logger.info(f"  - Average strength: {sum(s.get('strength_score', 0) for s in alpha_signals)/max(len(alpha_signals), 1):.0f}/100")

        logger.info(f"-"*60)
        logger.info(f"STRATEGIES EVALUATED: {len(best_strategies)} ranked")

        # Top strategy summary
        if best_strategies:
            top = best_strategies[0]
            logger.info(f"  BEST STRATEGY: {top['type']}")
            logger.info(f"  - Expected Return: {top.get('expected_return', 0):.2f}%")
            logger.info(f"  - Win Probability: {top.get('probability_profit', 0)*100:.1f}%")
            logger.info(f"  - Max Risk: ${top.get('max_risk', 0):.2f}")
            if 'sharpe' in top:
                logger.info(f"  - Sharpe Ratio: {top['sharpe']:.2f}")

        logger.info(f"="*60)

        return results

    def analyze_specific_opportunity(
        self,
        ticker: str,
        strike: float,
        expiration: datetime,
        option_type: str = 'call',
        option_premium: Optional[float] = None,
        data_source: str = 'yfinance'
    ) -> Dict:
        """
        Comprehensive analysis of a specific option opportunity

        Integrates:
        - Market context (price momentum, analyst targets, related assets)
        - Volatility analysis (IV vs HV)
        - Probability analysis (strike probabilities, expected value)
        - Decision engine (BUY/HOLD/AVOID recommendation)

        Args:
            ticker: Stock symbol
            strike: Option strike price
            expiration: Expiration date
            option_type: 'call' or 'put'
            option_premium: Option price (if known)
            data_source: Data source for analysis

        Returns:
            Comprehensive opportunity analysis with recommendation
        """
        logger.info(f"Analyzing specific opportunity: {ticker} ${strike} {option_type} exp {expiration.strftime('%Y-%m-%d')}")

        from data_manager import OptionsDataManager
        from market_context_analyzer import MarketContextAnalyzer
        from probability_framework import ProbabilityAnalyzer
        from decision_engine import DecisionEngine

        # Initialize components
        data_mgr = OptionsDataManager(cache_dir=self.cache_dir)
        market_analyzer = MarketContextAnalyzer()
        prob_analyzer = ProbabilityAnalyzer()
        decision_engine = DecisionEngine()

        # 1. Get market data
        current_price = data_mgr.get_current_price(ticker, data_source)
        historical_vol = data_mgr.calculate_historical_volatility(ticker, window=30)

        # 2. Market context analysis
        market_context = market_analyzer.analyze_comprehensive_context(ticker)

        # 3. Calculate time to expiry
        time_to_expiry = (expiration - datetime.now()).days / 365.0

        # 4. Get implied volatility from options chain if available
        implied_vol = historical_vol  # Default to HV
        if option_premium is None:
            # Try to find this specific option in the chain
            try:
                options_df = data_mgr.fetch_options_data(ticker, data_source)
                matching = options_df[
                    (options_df['strike'] == strike) &
                    (options_df['expiration'] == expiration.strftime('%Y-%m-%d')) &
                    (options_df['option_type'] == option_type)
                ]
                if not matching.empty:
                    option_premium = matching.iloc[0].get('mid') or matching.iloc[0].get('last', 0)
                    implied_vol = matching.iloc[0].get('impliedVolatility', historical_vol)
            except:
                pass

        # 5. Probability analysis
        probability_analysis = prob_analyzer.analyze_comprehensive_probabilities(
            current_price=current_price,
            strike=strike,
            time_to_expiry=time_to_expiry,
            volatility=implied_vol,
            option_premium=option_premium,
            option_type=option_type
        )

        # 6. Volatility environment
        volatility_analysis = {
            'implied_volatility': implied_vol,
            'historical_volatility': historical_vol,
            'iv_hv_spread': implied_vol - historical_vol
        }

        # 7. Decision engine evaluation
        decision = decision_engine.evaluate_opportunity(
            market_context=market_context,
            volatility_analysis=volatility_analysis,
            probability_analysis=probability_analysis
        )

        # Compile comprehensive results
        return {
            'ticker': ticker,
            'strike': strike,
            'expiration': expiration.isoformat(),
            'option_type': option_type,
            'option_premium': option_premium,

            'current_price': current_price,
            'historical_volatility': historical_vol,
            'implied_volatility': implied_vol,
            'time_to_expiry_days': int(time_to_expiry * 365),

            'market_context': market_context,
            'probability_analysis': probability_analysis,
            'volatility_analysis': volatility_analysis,
            'decision': decision,

            'timestamp': datetime.now().isoformat()
        }

    def analyze_multiple_tickers(
        self,
        tickers: List[str],
        data_source: str = 'yfinance',
        **kwargs
    ) -> pd.DataFrame:
        """
        Analyze multiple tickers and rank by opportunity

        Args:
            tickers: List of symbols
            data_source: Data source to use
            **kwargs: Additional arguments for analyze_ticker

        Returns:
            DataFrame with ranked opportunities across all tickers
        """
        logger.info(f"Analyzing {len(tickers)} tickers...")

        all_opportunities = []

        for ticker in tickers:
            try:
                result = self.analyze_ticker(ticker, data_source=data_source, **kwargs)

                if 'error' not in result:
                    # Extract top opportunities
                    if result['best_strategies']:
                        top_strategy = result['best_strategies'][0]
                        all_opportunities.append({
                            'ticker': ticker,
                            'strategy_type': top_strategy['type'],
                            'expected_return': top_strategy.get('expected_return', 0),
                            'probability_profit': top_strategy.get('probability_profit', 0),
                            'max_risk': top_strategy.get('max_risk', 0),
                            'sharpe_estimate': top_strategy.get('sharpe', 0),
                            'iv': top_strategy.get('implied_vol', 0),
                            'hv': result['historical_vol'],
                            'iv_hv_spread': top_strategy.get('implied_vol', 0) - result['historical_vol'],
                            'current_price': result['current_price']
                        })
            except Exception as e:
                logger.error(f"Failed to analyze {ticker}: {e}")
                continue

        # Create ranked DataFrame
        opportunities_df = pd.DataFrame(all_opportunities)

        if not opportunities_df.empty:
            # Rank by risk-adjusted return
            opportunities_df['rank_score'] = (
                opportunities_df['expected_return'] * opportunities_df['probability_profit']
                / (opportunities_df['max_risk'] + 1)
            )
            opportunities_df = opportunities_df.sort_values('rank_score', ascending=False)

        logger.info(f"Found {len(opportunities_df)} opportunities across {len(tickers)} tickers")

        return opportunities_df

    def generate_report(
        self,
        ticker: str,
        output_dir: Optional[str] = None,
        include_plots: bool = True
    ) -> str:
        """
        Generate comprehensive HTML report for a ticker

        Args:
            ticker: Ticker to report on
            output_dir: Output directory (default: cache_dir)
            include_plots: Whether to generate interactive plots

        Returns:
            Path to generated report
        """
        if ticker not in self.results:
            raise ValueError(f"No analysis results for {ticker}. Run analyze_ticker() first.")

        if output_dir is None:
            output_dir = self.cache_dir / ticker.upper() / "reports"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.results[ticker]

        # Generate report HTML
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_dir / f"{ticker}_analysis_{timestamp}.html"

        html_content = self._build_html_report(results, include_plots)

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Report generated: {report_file}")

        return str(report_file)

    def _build_html_report(self, results: Dict, include_plots: bool = True) -> str:
        """Build comprehensive HTML report with detailed analytics"""

        ticker = results['ticker']
        current_price = results['current_price']
        hv = results['historical_vol']
        alpha_signals = results['alpha_signals']
        strategies = results['best_strategies']
        options_df = results.get('options_data')
        seasonality_records = results.get('seasonality')

        # Calculate aggregate statistics
        total_signals = len(alpha_signals)
        high_conf_signals = sum(1 for s in alpha_signals if s.get('confidence') == 'HIGH')
        avg_signal_strength = sum(s.get('strength_score', 0) for s in alpha_signals) / max(total_signals, 1)

        # Volatility analysis
        iv_signals = [s for s in alpha_signals if 'implied_vol' in s]
        avg_iv = sum(s['implied_vol'] for s in iv_signals) / max(len(iv_signals), 1) if iv_signals else hv
        iv_hv_spread = avg_iv - hv

        # Optional figures (term structure, seasonality)
        ts_div = "<p>Not available.</p>"
        season_div = "<p>Not available.</p>"
        try:
            import pandas as pd  # noqa
            import plotly.graph_objects as go  # noqa
            if options_df is not None and not options_df.empty:
                dfc = options_df.copy()
                if not pd.api.types.is_datetime64_any_dtype(dfc['expiration']):
                    dfc['expiration'] = pd.to_datetime(dfc['expiration'])
                dfc['Tdays'] = (dfc['expiration'] - pd.Timestamp.now()).dt.days
                dfc['strike_diff'] = (dfc['strike'] - current_price).abs()
                atm = dfc.sort_values(['expiration','strike_diff']).groupby('expiration').head(2)
                term = atm.groupby('expiration')['impliedVolatility'].mean().reset_index()
                term = term[term['impliedVolatility']>0]
                if not term.empty:
                    term = term.sort_values('expiration')
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(x=term['expiration'], y=term['impliedVolatility']*100, mode='lines+markers', name='ATM IV'))
                    fig_ts.update_layout(title='ATM IV Term Structure', xaxis_title='Expiration', yaxis_title='IV (%)')
                    ts_div = fig_ts.to_html(include_plotlyjs='cdn', full_html=False)
                # SVI-Joint QC (optional) - disabled inline; keep import guard only
                try:
                    from svi_joint import fit_svi_joint  # noqa: F401
                    _ = fit_svi_joint
                except Exception:
                    pass
                # Heston calibration (optional)
                try:
                    from heston_calibration import calibrate_heston
                    # Build a light quotes set from ATM IV per expiry
                    quotes = []
                    atm = dfc.sort_values(['expiration','strike_diff']).groupby('expiration').head(1)
                    for _, row in atm.iterrows():
                        quotes.append({'K': float(row['strike']), 'T': float((row['expiration'] - pd.Timestamp.now()).days)/365.0, 'iv': float(row['impliedVolatility'])})
                    hcal = calibrate_heston(quotes, spot)
                    if hcal.get('status') == 'ok':
                        params = hcal.get('params',{})
                except Exception:
                    pass
            if seasonality_records:
                sdf = pd.DataFrame(seasonality_records)
                if 'expiration' in sdf.columns and 'z_seasonal' in sdf.columns:
                    sdf['expiration'] = pd.to_datetime(sdf['expiration'])
                    fig_season = go.Figure()
                    fig_season.add_trace(go.Bar(x=sdf['expiration'], y=sdf['z_seasonal'], name='Seasonality z'))
                    fig_season.update_layout(title='Seasonality z by Expiration (IV vs HV baseline)', xaxis_title='Expiration', yaxis_title='z-score')
                    season_div = fig_season.to_html(include_plotlyjs='cdn', full_html=False)
        except Exception:
            pass

        # Try to detect latest SVI outputs for quick links
        try:
            from glob import glob
            base = str(self.cache_dir / ticker.upper())
            params = sorted(glob(base + f"/{ticker}_svi_params_*.csv"))
            diags = sorted(glob(base + "/reports/" + f"{ticker}_svi_diagnostics_*.html"))
            # Monte Carlo links (if present)
            mc = sorted(glob(base + "/reports/" + f"{ticker}_mc_*.html"))
            if params or diags or mc:
                items = []
                if params:
                    items.append(f"<li>SVI params: {params[-1]}</li>")
                if diags:
                    items.append(f"<li>SVI diagnostics: {diags[-1]}</li>")
                if mc:
                    items.append(f"<li>Monte Carlo: {mc[-1]}</li>")
        except Exception:
            pass

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{ticker} Volatility Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .header h1 {{ margin: 0; font-size: 2.5em; }}
                .header p {{ margin: 5px 0; opacity: 0.9; }}
                .section {{ background: white; margin: 20px 0; padding: 25px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                .section h2 {{ color: #2c3e50; border-bottom: 3px solid #667eea; padding-bottom: 10px; margin-top: 0; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric {{ background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; border-radius: 8px; text-align: center; }}
                .metric-label {{ font-weight: 600; color: #555; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }}
                .metric-value {{ font-size: 2em; color: #2c3e50; font-weight: bold; margin-top: 10px; }}
                .metric-subtext {{ font-size: 0.85em; color: #777; margin-top: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
                th {{ background: #34495e; color: white; font-weight: 600; position: sticky; top: 0; }}
                tr:hover {{ background: #f8f9fa; }}
                .positive {{ color: #27ae60; font-weight: bold; }}
                .negative {{ color: #e74c3c; font-weight: bold; }}
                .neutral {{ color: #95a5a6; }}
                .badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; }}
                .badge-high {{ background: #27ae60; color: white; }}
                .badge-medium {{ background: #f39c12; color: white; }}
                .badge-low {{ background: #95a5a6; color: white; }}
                .interpretation {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 5px; }}
                .interpretation h3 {{ margin-top: 0; color: #856404; }}
                .greeks {{ font-family: 'Courier New', monospace; font-size: 0.9em; }}
                .detail-row {{ font-size: 0.9em; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{ticker} Comprehensive Volatility Analysis</h1>
                <p>Generated: {results['timestamp']}</p>
                <p>Data Source: <strong>{results['data_source'].upper()}</strong> | Analysis Engine: v2.0</p>
            </div>

            <div class="section">
                <h2>Market Overview & Key Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-label">Current Price</div>
                        <div class="metric-value">${current_price:.2f}</div>
                        <div class="metric-subtext">Spot price</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Historical Vol</div>
                        <div class="metric-value">{hv*100:.2f}%</div>
                        <div class="metric-subtext">30-day realized</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Implied Vol (Avg)</div>
                        <div class="metric-value">{avg_iv*100:.2f}%</div>
                        <div class="metric-subtext">Market pricing</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">IV-HV Spread</div>
                        <div class="metric-value {'positive' if iv_hv_spread > 0 else 'negative'}">{iv_hv_spread*100:+.2f}%</div>
                        <div class="metric-subtext">{'Expensive' if iv_hv_spread > 0 else 'Cheap' if iv_hv_spread < 0 else 'Fair'}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Options Analyzed</div>
                        <div class="metric-value">{results['n_options']}</div>
                        <div class="metric-subtext">Liquid contracts</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Alpha Signals</div>
                        <div class="metric-value">{total_signals}</div>
                        <div class="metric-subtext">{high_conf_signals} high confidence</div>
                    </div>
                </div>

                <div class="interpretation">
                    <h3>Market Interpretation</h3>
                    <p><strong>Volatility Assessment:</strong>
                    {"Implied volatility is EXPENSIVE relative to realized volatility. Market is pricing in more movement than historical data suggests. This creates a SELL VOLATILITY opportunity (short straddles, iron condors)." if iv_hv_spread > 0.05
                     else "Implied volatility is CHEAP relative to realized volatility. Market is not pricing in enough movement. This creates a BUY VOLATILITY opportunity (long straddles, strangles)." if iv_hv_spread < -0.05
                     else "Implied volatility is fairly priced relative to historical levels. No strong directional volatility bias."}</p>
                    <p><strong>Signal Quality:</strong> Average signal strength is {avg_signal_strength:.0f}/100 with {high_conf_signals}/{total_signals} high-confidence signals.</p>
                </div>
            </div>

            <div class="section">
                <h2>Alpha Signals Detected ({total_signals} total)</h2>
                <p style="color: #666; margin-bottom: 15px;">Signals are ranked by strength score (0-100). Higher scores indicate stronger opportunities.</p>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Signal Type</th>
                        <th>Action</th>
                        <th>Strength</th>
                        <th>Confidence</th>
                        <th>IV</th>
                        <th>HV</th>
                        <th>Spread</th>
                        <th>Details</th>
                    </tr>
        """

        for i, signal in enumerate(alpha_signals[:15], 1):  # Top 15
            iv = signal.get('implied_vol', avg_iv)
            spread = iv - hv
            spread_class = 'positive' if spread > 0.05 else 'negative' if spread < -0.05 else 'neutral'
            conf = signal.get('confidence', 'MEDIUM')
            badge_class = f"badge-{conf.lower()}"

            # Extract additional details
            details = []
            if 'expiration' in signal:
                details.append(f"Exp: {signal['expiration'].strftime('%Y-%m-%d') if hasattr(signal['expiration'], 'strftime') else signal['expiration']}")
            if 'strike' in signal:
                details.append(f"Strike: ${signal['strike']:.0f}")
            if 'straddle_price' in signal:
                details.append(f"Price: ${signal['straddle_price']:.2f}")
            if 'days_to_expiry' in signal:
                details.append(f"DTE: {signal['days_to_expiry']}d")

            details_str = " | ".join(details) if details else "See recommendation"

            html += f"""
                    <tr>
                        <td><strong>{i}</strong></td>
                        <td>{signal.get('type', 'N/A').replace('_', ' ')}</td>
                        <td><strong>{signal.get('signal', 'N/A').replace('_', ' ')}</strong></td>
                        <td><strong>{signal.get('strength_score', 0):.0f}/100</strong></td>
                        <td><span class="badge {badge_class}">{conf}</span></td>
                        <td>{iv*100:.2f}%</td>
                        <td>{hv*100:.2f}%</td>
                        <td class="{spread_class}"><strong>{spread*100:+.2f}%</strong></td>
                        <td class="detail-row">{details_str}</td>
                    </tr>
                    <tr>
                        <td colspan="9" style="background: #f8f9fa; font-size: 0.9em; padding: 8px 12px;">
                            <strong>Recommendation:</strong> {signal.get('recommendation', 'N/A')}
                        </td>
                    </tr>
            """

        html += """
                </table>
            </div>

            <div class="section">
                <h2>Top Ranked Strategies ({len(strategies)} evaluated)</h2>
                <p style="color: #666; margin-bottom: 15px;">Strategies ranked by risk-adjusted expected value. All metrics based on 10,000 Monte Carlo simulations.</p>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Strategy Type</th>
                        <th>Entry Cost</th>
                        <th>Expected Return</th>
                        <th>Win Probability</th>
                        <th>Max Risk</th>
                        <th>Sharpe</th>
                        <th>VaR (95%)</th>
                        <th>Greeks</th>
                    </tr>
        """

        for i, strategy in enumerate(strategies[:10], 1):
            exp_ret = strategy.get('expected_return', 0)
            ret_class = 'positive' if exp_ret > 5 else 'negative' if exp_ret < -5 else 'neutral'
            prob = strategy.get('probability_profit', 0)
            prob_class = 'positive' if prob > 0.6 else 'neutral' if prob > 0.5 else 'negative'

            max_risk = strategy.get('max_risk', 0)
            risk_str = f"${max_risk:.2f}" if max_risk != float('inf') else "Unlimited"

            # Format Greeks
            greeks_str = ""
            if 'greeks' in strategy:
                g = strategy['greeks']
                greeks_str = f"Î´:{g.get('delta', 0):.3f} Î³:{g.get('gamma', 0):.4f} Î¸:{g.get('theta', 0):.2f} Î½:{g.get('vega', 0):.2f}"

            html += f"""
                    <tr>
                        <td><strong>{i}</strong></td>
                        <td><strong>{strategy.get('type', 'N/A').replace('_', ' ')}</strong></td>
                        <td>${abs(strategy.get('entry_cost', 0)):.2f}</td>
                        <td class="{ret_class}"><strong>{exp_ret:+.2f}%</strong></td>
                        <td class="{prob_class}"><strong>{prob*100:.1f}%</strong></td>
                        <td>{risk_str}</td>
                        <td>{strategy.get('sharpe', 0):.2f}</td>
                        <td>${strategy.get('var_95', 0):.2f}</td>
                        <td class="greeks">{greeks_str}</td>
                    </tr>
            """

            # Add detail row
            detail_parts = []
            if 'expiration' in strategy:
                detail_parts.append(f"Expiration: {strategy['expiration'].strftime('%Y-%m-%d') if hasattr(strategy['expiration'], 'strftime') else strategy['expiration']}")
            if 'strike' in strategy:
                detail_parts.append(f"Strike: ${strategy['strike']:.0f}")
            if 'days_to_expiry' in strategy:
                detail_parts.append(f"DTE: {strategy['days_to_expiry']} days")
            if 'breakeven_up' in strategy and 'breakeven_down' in strategy:
                detail_parts.append(f"Breakevens: ${strategy['breakeven_down']:.2f} - ${strategy['breakeven_up']:.2f}")
            if 'expected_value' in strategy:
                detail_parts.append(f"Expected P&L: ${strategy['expected_value']:.2f}")

            if detail_parts or strategy.get('recommendation'):
                html += f"""
                    <tr>
                        <td colspan="9" style="background: #f8f9fa; font-size: 0.9em; padding: 8px 12px;">
                            {' | '.join(detail_parts)}
                            {('<br><strong>Note:</strong> ' + strategy.get('recommendation', '')) if strategy.get('recommendation') else ''}
                        </td>
                    </tr>
                """

        html += """
                </table>
            </div>

            <div class="section">
                <h2>Risk Disclaimer</h2>
                <p style="color: #666;">
                    This analysis is for educational purposes only. Options trading involves significant risk of loss.
                    Past volatility does not predict future volatility. Monte Carlo simulations use simplified assumptions
                    and may not reflect actual market conditions. Always paper trade first, use proper position sizing,
                    and consult a financial advisor before implementing any strategy.
                </p>
                <p style="color: #666; margin-top: 10px;">
                    <strong>Key Assumptions:</strong> Black-Scholes pricing model, lognormal price distribution,
                    constant volatility (not realistic), no transaction costs, no dividends (unless specified),
                    risk-neutral drift in simulations.
                </p>
            </div>

            <div class="section" style="background: #e8f5e9;">
                <h2>How to Use This Report</h2>
                <ol style="line-height: 1.8;">
                    <li><strong>Check IV-HV Spread:</strong> Positive spread suggests selling volatility, negative suggests buying</li>
                    <li><strong>Review Alpha Signals:</strong> Focus on HIGH confidence signals with strength > 70</li>
                    <li><strong>Analyze Top Strategies:</strong> Look for win probability > 60% and positive expected return</li>
                    <li><strong>Understand Greeks:</strong> Delta (directional exposure), Gamma (acceleration), Theta (time decay), Vega (vol exposure)</li>
                    <li><strong>Consider Risk:</strong> Never risk more than you can afford to lose, especially on unlimited risk strategies</li>
                    <li><strong>Backtest First:</strong> Paper trade any strategy before using real capital</li>
                </ol>
            </div>
        </body>
        </html>
        """

        return html

    def export_results(self, ticker: str, format: str = 'csv') -> str:
        """
        Export analysis results

        Args:
            ticker: Ticker to export
            format: 'csv' or 'json'

        Returns:
            Path to exported file
        """
        if ticker not in self.results:
            raise ValueError(f"No results for {ticker}")

        results = self.results[ticker]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if format == 'csv':
            filepath = self.cache_dir / f"{ticker}_results_{timestamp}.csv"
            results['options_data'].to_csv(filepath, index=False)
        elif format == 'json':
            filepath = self.cache_dir / f"{ticker}_results_{timestamp}.json"
            import json
            with open(filepath, 'w') as f:
                # Remove non-serializable objects
                export_data = {
                    'ticker': results['ticker'],
                    'timestamp': results['timestamp'],
                    'current_price': results['current_price'],
                    'historical_vol': results['historical_vol'],
                    'alpha_signals': results['alpha_signals'],
                    'best_strategies': results['best_strategies']
                }
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Results exported to {filepath}")
        return str(filepath)

    def generate_commodity_report(self, ticker: str) -> str:
        """
        Generate an interactive HTML commodity analytics report with term structure,
        roll yield, and seasonality context in the same style as the main report.
        """
        from commodity_analyzer import CommodityAnalyzer
        from data_manager import OptionsDataManager
        PLOT_OK = True
        try:
            import plotly.graph_objects as go  # type: ignore
            import pandas as pd  # type: ignore
        except Exception:
            PLOT_OK = False

        dm = OptionsDataManager(cache_dir=self.cache_dir)
        df = dm.fetch_options_data(ticker, data_source='yfinance')
        price = dm.get_current_price(ticker)

        ca = CommodityAnalyzer()
        res = ca.analyze_commodity(ticker, current_price=price, options_df=df)

        # Build plots
        fig_ts = None
        if PLOT_OK and res.get('term_structure', {}).get('status') == 'success':
            term_df = pd.DataFrame(res['term_structure']['term_data'])
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=term_df['days_to_expiry'],
                y=term_df['avg_iv'] * 100,
                mode='lines+markers',
                name='ATM IV',
            ))
            fig_ts.update_layout(
                title='ATM Term Structure', xaxis_title='Days to Expiry', yaxis_title='IV (%)'
            )

        fig_season = None
        if PLOT_OK and res.get('seasonality', {}).get('status') == 'success':
            season = res['seasonality']
            peak = season.get('peak_months', [])
            trough = season.get('trough_months', [])
            months = list(range(1,13))
            vals = [(2 if m in peak else 1 if m in trough else 0) for m in months]
            fig_season = go.Figure()
            fig_season.add_bar(x=months, y=vals, marker_color=['crimson' if v==2 else 'steelblue' if v==1 else 'lightgray' for v in vals])
            fig_season.update_layout(title='Seasonality Map (2=Peak, 1=Trough)', xaxis_title='Month', yaxis_title='Seasonality Level')

        # Compose HTML using the main report style
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = self.cache_dir / ticker.upper() / 'reports'
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"{ticker}_commodity_report_{ts}.html"

        def fig_div(fig):
            if not PLOT_OK:
                return '<p>Plots unavailable (plotly/pandas not installed).</p>'
            return fig.to_html(include_plotlyjs='cdn', full_html=False) if fig is not None else '<p>No data.</p>'

        term = res.get('term_structure', {})
        roll = res.get('roll_yield', {})
        season = res.get('seasonality', {})

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{ticker} Commodity Analytics</title>
            <meta charset="utf-8" />
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }}
                .header {{ background: linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%); color: white; padding: 30px; border-radius: 10px; }}
                .header h1 {{ margin: 0; font-size: 2.2em; }}
                .section {{ background: white; margin: 20px 0; padding: 25px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                .section h2 {{ color: #2c3e50; border-bottom: 3px solid #4ca1af; padding-bottom: 10px; margin-top: 0; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric {{ background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; border-radius: 8px; text-align: center; }}
                .metric-label {{ font-weight: 600; color: #555; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }}
                .metric-value {{ font-size: 1.8em; color: #2c3e50; font-weight: bold; margin-top: 10px; }}
                .small {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{ticker} Commodity Analytics</h1>
                <p>Generated: {datetime.now().isoformat()}</p>
            </div>

            <div class="section">
                <h2>Seasonality & Term Structure</h2>
                <p class="neutral">Seasonality adjusts IV for cyclical patterns (e.g., harvest/planting in commodities). z > +1 indicates seasonally elevated IV; z < -1 seasonally depressed IV.</p>
                <div style="margin-top:10px;">{fig_div(fig_ts)}</div>
                <div style="margin-top:20px;">{fig_div(fig_season)}</div>
            </div>

            <div class="section">
                <h2>Overview</h2>
                <div class="metrics-grid">
                    <div class="metric"><div class="metric-label">Price</div><div class="metric-value">${price:.2f}</div><div class="small">Spot</div></div>
                    <div class="metric"><div class="metric-label">Term Structure</div><div class="metric-value">{term.get('structure_type','N/A')}</div><div class="small">{term.get('description','')}</div></div>
                    <div class="metric"><div class="metric-label">Roll Yield</div><div class="metric-value">{roll.get('roll_yield_annual_pct',0):+.1f}%</div><div class="small">{roll.get('impact','').title()}</div></div>
                    <div class="metric"><div class="metric-label">Seasonality Phase</div><div class="metric-value">{season.get('current_phase','N/A')}</div><div class="small">{season.get('description','')}</div></div>
                </div>
            </div>

            <div class="section">
                <h2>ATM IV Term Structure</h2>
                {fig_div(fig_ts)}
            </div>

            <div class="section">
                <h2>Seasonality</h2>
                {fig_div(fig_season)}
                <p class="small">Bias: {res.get('seasonality_iv_bias','neutral')}. {res.get('seasonality_adjusted_iv_comment','')}</p>
            </div>

            <div class="section">
                <h2>Trading Implications</h2>
                <ul>
        """
        for rec in res.get('recommendations', []):
            html += f"<li>{rec}</li>"
        html += """
                </ul>
            </div>
        </body>
        </html>
        """

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"Commodity report saved to {filepath}")
        return str(filepath)

    def generate_mc_report(self, ticker: str, model: str, summary: Dict, paths_csv_path: str) -> str:
        """
        Generate interactive Monte Carlo report with sample paths and terminal distribution.
        """
        try:
            import numpy as np
            import plotly.graph_objects as go
        except Exception:
            raise RuntimeError("Plotly and numpy required for Monte Carlo report")

        # Load sample paths
        try:
            paths = np.loadtxt(paths_csv_path, delimiter=',')
        except Exception:
            paths = None

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = self.cache_dir / ticker.upper() / 'reports'
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"{ticker}_mc_{model.lower()}_report_{ts}.html"

        def fig_div(fig):
            return fig.to_html(include_plotlyjs='cdn', full_html=False) if fig is not None else '<p>No data.</p>'

        # Build figures
        fig_paths = None
        fig_hist = None
        if paths is not None and paths.size > 0:
            x = list(range(paths.shape[1]))
            fig_paths = go.Figure()
            for i in range(min(25, paths.shape[0])):
                fig_paths.add_trace(go.Scatter(x=x, y=paths[i, :], mode='lines', line=dict(width=1), opacity=0.6, showlegend=False))
            fig_paths.update_layout(title='Sample Price Paths', xaxis_title='Step', yaxis_title='Price')

            terminal = paths[:, -1]
            ret = terminal / paths[:, 0] - 1.0
            fig_hist = go.Figure(data=[go.Histogram(x=ret, nbinsx=40)])
            fig_hist.update_layout(title='Terminal Return Distribution', xaxis_title='Return', yaxis_title='Frequency')

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{ticker} Monte Carlo - {model}</title>
            <meta charset="utf-8" />
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }}
                .header {{ background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%); color: white; padding: 30px; border-radius: 10px; }}
                .header h1 {{ margin: 0; font-size: 2.2em; }}
                .section {{ background: white; margin: 20px 0; padding: 25px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                .section h2 {{ color: #2c3e50; border-bottom: 3px solid #2c5364; padding-bottom: 10px; margin-top: 0; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; }}
                .metric {{ background: #eef2f3; padding: 16px; border-radius: 8px; }}
                .label {{ font-size: 0.9em; color: #666; text-transform: uppercase; }}
                .value {{ font-size: 1.4em; font-weight: 600; color: #2c3e50; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{ticker} Monte Carlo ({model})</h1>
                <p>Generated: {summary.get('timestamp','')}</p>
                <p>{summary.get('explanation','')}</p>
            </div>

            <div class="section">
                <h2>Summary</h2>
                <div class="grid">
                    <div class="metric"><div class="label">S0</div><div class="value">{summary.get('S0',0):.2f}</div></div>
                    <div class="metric"><div class="label">S_T Mean</div><div class="value">{summary.get('S_T_mean',0):.2f}</div></div>
                    <div class="metric"><div class="label">S_T Std</div><div class="value">{summary.get('S_T_std',0):.2f}</div></div>
                    <div class="metric"><div class="label">Mean Return</div><div class="value">{summary.get('ret_mean',0)*100:.2f}%</div></div>
                    <div class="metric"><div class="label">Return Std</div><div class="value">{summary.get('ret_std',0)*100:.2f}%</div></div>
                    <div class="metric"><div class="label">P(Up)</div><div class="value">{summary.get('p_up',0)*100:.1f}%</div></div>
                    <div class="metric"><div class="label">P(<= -10%)</div><div class="value">{summary.get('p_10pct_drop',0)*100:.1f}%</div></div>
                </div>
            </div>

            <div class="section"><h2>Sample Paths</h2>{fig_div(fig_paths)}</div>
            <div class="section"><h2>Terminal Return Histogram</h2>{fig_div(fig_hist)}</div>
        </body>
        </html>
        """

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"Monte Carlo report saved to {filepath}")
        return str(filepath)


def interactive_menu():
    """Interactive command-line menu"""

    print("=" * 80)
    print("VOLATILITY ANALYSIS SYSTEM")
    print("=" * 80)

    # Guided paths to help users pick an analysis workflow (covers all options)
    print("\nGUIDED PATHS (What to run and why)")
    print("-" * 80)
    print("Visualization (1â€“4): Quick visuals and dashboards")
    print("  1) 3D IV Surface: Build an implied vol surface for a ticker and save" 
          " an interactive HTML (â€¦/data_cache/vol_surface/{TICKER}/reports). Input: ticker.")
    print("  2) Payoff Diagram: Manually enter strategy legs to visualize payoff at expiry." 
          " Outputs HTML (or CSV fallback). Inputs: S0 and legs.")
    print("  3) Backtest Visualizer: Render an equity curve from a CSV (e.g., from option 13)." 
          " Input: equity CSV path. Output: HTML in â€¦/{TICKER}/reports.")
    print("  4) Dashboard: Launch Streamlit summary (latest SVI/MC artifacts detected automatically).")

    print("\nAnalytics (5â€“13): Full analysis, signals, strategies, and research tools")
    print("  5) Single Ticker Analysis: Fetch options, compute HV & seasonality, screen signals"
          " (incl. calendars/diagonals), rank strategies. Then run 7 to save the report.")
    print("     Inputs: ticker (+ data source). Outputs: inâ€‘memory results; perâ€‘ticker caches under â€¦/{TICKER}/.")
    print("  6) Multiâ€‘Ticker Screener: Analyze a list of tickers and rank top opportunities across them."
          " Input: commaâ€‘separated tickers. Output: screener_results_â€¦csv under â€¦/vol_surface.")
    print("  7) Generate HTML Report: Save a full interactive report for a ticker analyzed with 5," 
          " including: Market metrics, Alpha signals, Seasonality & Term Structure (with plots),"
          " and links to SVI/MC diagnostics. Output: â€¦/{TICKER}/reports/{TICKER}_analysis_â€¦html.")
    print("  8) Specific Option Opportunity: Deep dive on a single option (strike/expiry/type)"
          " with context, probability, decision engine. Output: JSON in â€¦/{TICKER}/.")
    print("  9) Commodity Analytics: Seasonalityâ€‘aware term structure and rollâ€‘yield assessment"
          " (e.g., GLD/USO/UNG/WEAT/SOYB/PPLT/PALL). Output: HTML in â€¦/{TICKER}/reports.")
    print(" 10) SVI Surface Fitting: Fit arbitrageâ€‘aware SVI per expiry; save params CSV and a diagnostics HTML"
          " (market smiles vs fits). Output: CSV in â€¦/{TICKER}/ and HTML in â€¦/{TICKER}/reports.")
    print(" 11) Monte Carlo Simulation: Run BS/Merton/Heston; save an interactive report"
          " with sample paths and terminal distribution. Output: HTML in â€¦/{TICKER}/reports.")
    print(" 12) Interactive Strategy Builder: Guided construction of strategies with optional tickerâ€‘based"
          " ATM premium suggestions; outputs payoff and (optional) MC HTML files.")
    print(" 13) Strategy Backtest: Simplified historical backtest (yfinanceâ€‘based) for a chosen template;"
          " outputs equity CSV and an equity curve HTML.")

    print("\nUtilities (14â€“16): Export and validation")
    print(" 14) Export Results: Save the current analysis (from 5) to CSV/JSON under â€¦/{TICKER}/.")
    print(" 15) Validate BS Pricing: Compare Blackâ€“Scholes to QuantLib (if installed) for a European option."
          " Input: ticker, strike, days, type.")
    print(" 16) Exit: Quit the application.")

    print("\nQuick start: For a full deep dive, run 5 (analyze) â†’ 7 (report). For commodities, run 9.")

    system = VolatilityAnalysisSystem()

    def _usage_tip(opt: str):
        tips = {
            '1': (
                "3D IV Surface: Use to spot smiles/skews and term shape quickly. "
                "P&L: Identifies rich/cheap regions (e.g., expensive wings) to sell or buy; "
                "helps choose strikes/maturities that align with edge and avoid convexity traps."
            ),
            '2': (
                "Payoff Diagram: Sanityâ€‘check multiâ€‘leg structures before trading. P&L: "
                "Visualizes breakevens/max loss; prevents constructing positions with hidden tail risks; "
                "optimizes strike selection to improve expectancy."
            ),
            '3': (
                "Backtest Visualizer: Turn equity CSVs into an equity curve. P&L: "
                "Validates strategy persistence and draws attention to drawdowns, reducing overâ€‘sizing and timing mistakes."
            ),
            '4': (
                "Dashboard: Overview of latest SVI/MC artifacts. P&L: "
                "Faster situational awareness â†’ quicker pivots from long to short vol (and vice versa)."
            ),
            '5': (
                "Single Ticker Analysis: Full IV/HV + seasonality + strategy ranking. P&L: "
                "Buys vol when cheap seasonally and realized supports; sells when rich and term carry backs it; "
                "reduces false signals by factoring cyclicality and liquidity. After this, run 7 to save report."
            ),
            '6': (
                "Multiâ€‘Ticker Screener: Surfaces the best opportunities across a list. P&L: "
                "Concentrates capital in the highest EV ideas, improving portfolio edge per unit risk."
            ),
            '7': (
                "Generate HTML Report: Persists a rich, auditable analysis with plots and links. P&L: "
                "Forces disciplined review (seasonality, term, signals) to avoid impulsive entries; "
                "improves decision quality and repeatability."
            ),
            '8': (
                "Specific Option Opportunity: Deepâ€‘dives a single contract. P&L: "
                "Quantifies strike reach/probability and breakevens; reduces overpaying for tails and underâ€‘hedging core risk."
            ),
            '9': (
                "Commodity Analytics: Seasonality + term structure + roll yield. P&L: "
                "Avoids selling â€˜falseâ€™ expensive IV during harvest/roll; captures positive carry via calendars/diagonals."
            ),
            '10': (
                "SVI Surface Fitting: Arbitrageâ€‘aware surface and diagnostics. P&L: "
                "Cleaner Greeks/prices â†’ better hedges; identifies skew/term mispricings for RR/butterflies/calendars."
            ),
            '11': (
                "Monte Carlo (BS/Merton/Heston): Path/terminal scenarios. P&L: "
                "Sizes positions to distribution tails; selects models to match regime (jumps vs stochastic vol) â†’ fewer leftâ€‘tail losses."
            ),
            '12': (
                "Interactive Strategy Builder: Construct/test strategies with suggested ATM premiums. P&L: "
                "Prevents hidden unlimitedâ€‘risk structures; aligns Greeks with thesis; improves breakeven placement."
            ),
            '13': (
                "Strategy Backtest: Simplified historical validation. P&L: "
                "Removes overfit ideas before capital; informs realistic win rate/expectancy and drawdowns."
            ),
            '14': (
                "Export Results: Persist CSV/JSON for tracking and research. P&L: "
                "Enables postâ€‘trade analysis to refine rules and cut losing patterns sooner."
            ),
            '15': (
                "Validate BS Pricing (QuantLib): Crossâ€‘check pricing. P&L: "
                "Reduces model error slippage that can silently erode edge, especially for wings/long maturities."
            ),
        }
        msg = tips.get(opt)
        if msg:
            print("\nUSAGE & P&L RATIONALE")
            print("-" * 80)
            print(msg)
            print("-" * 80)

    while True:
        print("\n" + "=" * 80)
        print("MAIN MENU")
        print("=" * 80)
        print("VISUALIZATION")
        print("1. 3D Volatility Surface - IV surface plot [plotly_visualizer.py, data_manager.py]")
        print("2. Payoff Diagram - manual strategy legs [enhanced_visualizer.py]")
        print("3. Backtest Visualizer - equity curve [backtest_visualizer.py]")
        print("4. Dashboard - Streamlit UI [streamlit_dashboard.py]")
        print()
        print("ANALYTICS")
        print("5. Single Ticker Analysis - IV/HV, signals, strategies [data_manager.py, alpha_screener.py, strategy_optimizer.py, vol_arbitrage.py, greeks_calculator.py]")
        print("6. Multi-Ticker Screener - ranked opportunities [vol_analysis_system.py]")
        print("7. Generate HTML Report - interactive analysis [vol_analysis_system.py]")
        print("8. Specific Option Opportunity - context + decision [data_manager.py, market_context_analyzer.py, probability_framework.py, decision_engine.py]")
        print("9. Commodity Metrics - contango/roll/seasonality [commodity_analyzer.py, data_manager.py]")
        print("10. SVI Surface Fitting - per expiry [svi_surface_fitter.py]")
        print("11. Monte Carlo Simulation - BS/Merton/Heston [enhanced_monte_carlo.py]")
        print("12. Interactive Strategy Builder - templates + MC [interactive_strategy_builder.py, payoff_scenario_analysis.py]")
        print("13. Strategy Backtest - simplified P&L [strategy_backtester.py]")
        print()
        print("UTILITIES")
        print("14. Export Results - CSV/JSON [vol_analysis_system.py]")
        print("15. Validate BS Pricing (QuantLib, optional) [quantlib_validation.py]")
        print("16. Exit")

        choice = input("\nSelect option (1-16). Tips appear as you choose: ").strip()
        _usage_tip(choice)

        if choice == '1':
            # 3D Volatility Surface
            try:
                from data_manager import OptionsDataManager
                from plotly_visualizer import VolatilitySurfaceVisualizer
                print("\nBuild 3D IV Surface for a ticker (yfinance)")
                ticker = input("Ticker (e.g., SPY): ").strip().upper()
                dmgr = OptionsDataManager(cache_dir=system.cache_dir)
                df = dmgr.fetch_options_data(ticker, data_source='yfinance')
                if df.empty or 'impliedVolatility' not in df.columns:
                    print("No options IV data available.")
                else:
                    dfc = df.copy()
                    dfc['T'] = (dfc['expiration'] - pd.Timestamp.now()).dt.days / 365.0
                    dfc = dfc[dfc['T'] > 0]
                    piv = dfc.pivot_table(index='strike', columns='T', values='impliedVolatility', aggfunc='mean')
                    strikes = piv.index.values
                    maturities = piv.columns.values
                    iv_surface = piv.values
                    reports = system.cache_dir / ticker.upper() / 'reports'
                    reports.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    out_html = reports / f"{ticker}_iv_surface_{ts}.html"
                    viz = VolatilitySurfaceVisualizer(theme='plotly_white')
                    viz.plot_3d_surface(strikes, maturities, iv_surface, spot_price=dmgr.get_current_price(ticker), save_html=str(out_html))
                    print(f"Saved IV surface to: {out_html}")
            except Exception as e:
                print(f"\nError generating IV surface: {e}")

        elif choice == '2':
            # Payoff Diagram
            try:
                from enhanced_visualizer import payoff_diagram
                s0_input = input("Underlying price S0 (e.g., 100): ").strip()
                S0 = float(s0_input)
                legs = []
                print("Enter option legs (empty type to finish):")
                while True:
                    typ = input("  type (call/put): ").strip().lower()
                    if not typ:
                        break
                    strike = float(input("  strike: ").strip())
                    qty = float(input("  qty (+1 long, -1 short): ").strip())
                    prem_in = input("  premium (optional, default 0): ").strip()
                    prem = float(prem_in) if prem_in else 0.0
                    legs.append({'type': typ, 'strike': strike, 'qty': qty, 'premium': prem})
                out = payoff_diagram(S0, legs)
                print(f"\nPayoff saved to: {out}")
            except Exception as e:
                print(f"\nError generating payoff: {e}")

        elif choice == '3':
            # Backtest Visualizer
            try:
                from backtest_visualizer import plot_equity
                path = input("Path to equity CSV: ").strip()
                out = plot_equity(path)
                print(f"\nEquity visualization saved to: {out}")
            except Exception as e:
                print(f"\nError visualizing equity: {e}")

        elif choice == '4':
            # Streamlit Dashboard
            try:
                print("\nAttempting to launch Streamlit dashboard (CTRL+C to stop):")
                print("If it does not open automatically, run:")
                print("  streamlit run Finance_Trading_Coding/Vol_Surface/streamlit_dashboard.py")
                import subprocess, sys
                subprocess.Popen([
                    sys.executable, '-m', 'streamlit', 'run',
                    str(Path(__file__).with_name('streamlit_dashboard.py'))
                ])
            except Exception as e:
                print(f"\nCould not launch Streamlit: {e}")

        elif choice == '5':
            ticker = input("Enter ticker symbol: ").strip().upper()

            print("\nSelect data source:")
            print("1. Yahoo Finance (free, public)")
            print("2. IBKR (requires connection)")
            data_choice = input("Choice (1 or 2): ").strip()

            data_source = 'yfinance' if data_choice == '1' else 'ibkr'

            sec_type = 'STK'
            exchange = 'SMART'

            if data_source == 'ibkr':
                sec_type = input("Security type (STK/FUT, default STK): ").strip().upper() or 'STK'
                exchange = input("Exchange (default SMART): ").strip().upper() or 'SMART'

            print(f"\nAnalyzing {ticker}...")

            try:
                result = system.analyze_ticker(
                    ticker=ticker,
                    data_source=data_source,
                    sec_type=sec_type,
                    exchange=exchange
                )

                if 'error' in result:
                    print(f"\nError: {result['error']}")
                else:
                    print(f"\nAnalysis complete!")
                    print(f"Current Price: ${result['current_price']:.2f}")
                    print(f"Historical Vol: {result['historical_vol']*100:.2f}%")
                    print(f"Alpha Signals Found: {len(result['alpha_signals'])}")
                    print(f"Strategies Evaluated: {len(result['best_strategies'])}")

                    if result['best_strategies']:
                        print("\nTop 3 Strategies:")
                        for i, strat in enumerate(result['best_strategies'][:3], 1):
                            print(f"  {i}. {strat['type']}: Exp Return ${strat.get('expected_return', 0):.2f}, "
                                  f"Prob {strat.get('probability_profit', 0)*100:.1f}%")

            except Exception as e:
                print(f"\nError during analysis: {e}")
                import traceback
                traceback.print_exc()

        elif choice == '6':
            tickers_input = input("Enter ticker symbols (comma-separated): ").strip()
            tickers = [t.strip().upper() for t in tickers_input.split(',')]

            print("\nSelect data source:")
            print("1. Yahoo Finance")
            print("2. IBKR")
            data_choice = input("Choice (1 or 2): ").strip()
            data_source = 'yfinance' if data_choice == '1' else 'ibkr'

            print(f"\nAnalyzing {len(tickers)} tickers...")

            try:
                opportunities = system.analyze_multiple_tickers(tickers, data_source=data_source)

                if not opportunities.empty:
                    print("\n" + "=" * 80)
                    print("TOP OPPORTUNITIES")
                    print("=" * 80)
                    print(opportunities.to_string(index=False))

                    # Save to file
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filepath = system.cache_dir / f"screener_results_{timestamp}.csv"
                    opportunities.to_csv(filepath, index=False)
                    print(f"\nResults saved to: {filepath}")
                else:
                    print("\nNo opportunities found.")

            except Exception as e:
                print(f"\nError during screening: {e}")
                import traceback
                traceback.print_exc()

        elif choice == '7':
            if not system.results:
                print("\nNo analyzed tickers. Run analysis first (option 1 or 2).")
                continue

            print("\nAnalyzed tickers:", ', '.join(system.results.keys()))
            ticker = input("Enter ticker to generate report for: ").strip().upper()

            if ticker not in system.results:
                print(f"\n{ticker} not found in analyzed tickers.")
                continue

            try:
                report_path = system.generate_report(ticker)
                print(f"\nReport generated: {report_path}")
                print("Open this file in your browser to view the interactive report.")
            except Exception as e:
                print(f"\nError generating report: {e}")

        elif choice == '14':
            if not system.results:
                print("\nNo analyzed tickers. Run analysis first.")
                continue

            print("\nAnalyzed tickers:", ', '.join(system.results.keys()))
            ticker = input("Enter ticker to export: ").strip().upper()

            if ticker not in system.results:
                print(f"\n{ticker} not found.")
                continue

            print("\nExport format:")
            print("1. CSV")
            print("2. JSON")
            format_choice = input("Choice (1 or 2): ").strip()
            export_format = 'csv' if format_choice == '1' else 'json'

            try:
                filepath = system.export_results(ticker, format=export_format)
                print(f"\nExported to: {filepath}")
            except Exception as e:
                print(f"\nError exporting: {e}")

        elif choice == '16':
            print("\nExiting...")
            break

        elif choice == '10':
            try:
                ticker = input("Enter ticker symbol: ").strip().upper()
                print("\nData source: using Yahoo Finance by default (per project setting). SVI fits per expiry; diagnostics report will be saved under {TICKER}/reports.\n")
                fitter = SVISurfaceFitter()
                out_path = fitter.fit_ticker(ticker, data_source='yfinance')
                print(f"\nSVI parameters saved to: {out_path}")
            except Exception as e:
                print(f"\nError fitting SVI: {e}")

        elif choice == '11':
            try:
                ticker = input("Enter ticker symbol: ").strip().upper()
                print("\nSelect model: (BS: constant vol; Merton: jumps; Heston: stochastic vol)")
                print("1. Blackâ€“Scholes (constant vol)")
                print("2. Merton Jumpâ€“Diffusion")
                print("3. Heston (stochastic vol)")
                m_choice = input("Choice (1-3): ").strip()
                model_map = {'1': 'BS', '2': 'MERTON', '3': 'HESTON'}
                model = model_map.get(m_choice, 'BS')

                # Static explanation (2-3 lines)
                explain = MODEL_EXPLANATIONS.get(model, "")
                print("\nModel:", model)
                print("Explanation:", explain)
                if model == 'MERTON':
                    print("Params: lambda (jump intensity), muJ (mean jump), sigmaJ (jump vol)")
                elif model == 'HESTON':
                    print("Params: v0, kappa, theta, xi, rho (variance process)")

                days = input("Horizon in days (default 30): ").strip()
                days = int(days) if days else 30
                paths = input("Number of paths (default 5000): ").strip()
                paths = int(paths) if paths else 5000

                params = {}
                if model == 'MERTON':
                    lam = input("lambda (0.2): ").strip()
                    muJ = input("muJ (-0.05): ").strip()
                    sigmaJ = input("sigmaJ (0.10): ").strip()
                    if lam: params['lambda'] = float(lam)
                    if muJ: params['muJ'] = float(muJ)
                    if sigmaJ: params['sigmaJ'] = float(sigmaJ)
                elif model == 'HESTON':
                    v0 = input("v0 (hv^2 default): ").strip()
                    kappa = input("kappa (2.0): ").strip()
                    theta = input("theta (hv^2 default): ").strip()
                    xi = input("xi (0.5): ").strip()
                    rho = input("rho (-0.5): ").strip()
                    if v0: params['v0'] = float(v0)
                    if kappa: params['kappa'] = float(kappa)
                    if theta: params['theta'] = float(theta)
                    if xi: params['xi'] = float(xi)
                    if rho: params['rho'] = float(rho)

                sim = MonteCarloSimulator()
                result = sim.run_ticker(ticker, model=model, horizon_days=days, paths=paths, data_source='yfinance', params=params)
                print("\nMonte Carlo summary:")
                for k, v in result['summary'].items():
                    print(f"  {k}: {v}")
                mc_html = self.generate_mc_report(ticker, model, result['summary'], result['paths_csv_path'])
                print(f"Monte Carlo report saved to: {mc_html}")
            except Exception as e:
                print(f"\nError running Monte Carlo: {e}")
        elif choice == '13':
            try:
                from strategy_backtester import StrategyBacktester
                from backtest_visualizer import plot_equity
                ticker = input("Ticker (e.g., SPY): ").strip().upper()
                strat = input("Strategy (STRADDLE/STRANGLE/CALENDAR/BUTTERFLY): ").strip().upper() or 'STRADDLE'
                start = input("Start date (YYYY-MM-DD): ").strip()
                end = input("End date (YYYY-MM-DD): ").strip()
                bt = StrategyBacktester()
                # Minimal default rule sets
                entry = {'vol_threshold': 0.2}
                exit_rules = {'days_hold': 20}
                res = bt.backtest_strategy(ticker, strat, entry, exit_rules, start, end)
                # Save equity
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                eq_path = self.cache_dir / f"{ticker}_{strat}_equity_{ts}.csv"
                if not res['equity_curve'].empty:
                    res['equity_curve'].to_csv(eq_path, index=False)
                    print(f"\nPerformance: {res['performance']}")
                    print(f"Equity CSV: {eq_path}")
                    try:
                        html = plot_equity(str(eq_path), output_dir=str(self.cache_dir))
                        print(f"Equity plot: {html}")
                    except Exception:
                        pass
                else:
                    print("\nNo trades or equity curve generated.")
            except Exception as e:
                print(f"\nError running strategy backtest: {e}")

        elif choice == '12':
            try:
                from interactive_strategy_builder import interactive_strategy_builder as isb
                t = input("Enter ticker for strategy builder (optional): ").strip().upper()
                isb(ticker=t or None, data_source='yfinance')
            except Exception as e:
                print(f"\nError launching Interactive Strategy Builder: {e}")

        elif choice == '15':
            try:
                from quantlib_validation import validate_european_price
                ticker = input("Ticker (e.g., SPY): ").strip().upper()
                K = float(input("Strike (e.g., 100): ").strip())
                days = int(input("Days to expiry (e.g., 30): ").strip())
                opt_type = (input("Option type (call/put): ").strip().lower() or 'call')
                from data_manager import OptionsDataManager
                dm = OptionsDataManager(cache_dir=system.cache_dir)
                S = dm.get_current_price(ticker)
                T = max(1, days) / 365.0
                hv = dm.calculate_historical_volatility(ticker, window=30)
                res = validate_european_price(S=S, K=K, T=T, sigma=hv, r=0.0, q=0.0, option_type=opt_type)
                print("\nValidation result:")
                for k,v in res.items():
                    print(f"  {k}: {v}")
            except Exception as e:
                print(f"\nError during QuantLib validation: {e}")

        elif choice == '8':
            try:
                ticker = input("Ticker (e.g., SPY): ").strip().upper()
                strike = float(input("Strike (e.g., 100): ").strip())
                exp_str = input("Expiration (YYYY-MM-DD): ").strip()
                from datetime import datetime as _dt
                expiration = _dt.strptime(exp_str, '%Y-%m-%d')
                opt_type = (input("Option type (call/put, default call): ").strip().lower() or 'call')
                premium_in = input("Option premium (optional): ").strip()
                premium = float(premium_in) if premium_in else None
                result = system.analyze_specific_opportunity(
                    ticker=ticker,
                    strike=strike,
                    expiration=expiration,
                    option_type=opt_type,
                    option_premium=premium,
                    data_source='yfinance'
                )
                # Save to JSON
                import json
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                out = system.cache_dir / f"{ticker}_opportunity_{ts}.json"
                with open(out, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"Saved opportunity analysis to: {out}")
            except Exception as e:
                print(f"\nError analyzing specific opportunity: {e}")

        elif choice == '9':
            try:
                from data_manager import OptionsDataManager
                from commodity_analyzer import CommodityAnalyzer
                ticker = input("Commodity ETF (e.g., USO, UNG, GLD): ").strip().upper()
                dmgr = OptionsDataManager(cache_dir=system.cache_dir)
                df = dmgr.fetch_options_data(ticker, data_source='yfinance')
                price = dmgr.get_current_price(ticker)
                ca = CommodityAnalyzer()
                res = ca.analyze_commodity(ticker, current_price=price, options_df=df)
                import json
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                out = system.cache_dir / f"{ticker}_commodity_metrics_{ts}.json"
                with open(out, 'w') as f:
                    json.dump(res, f, indent=2, default=str)
                print(f"Saved commodity metrics to: {out}")
            except Exception as e:
                print(f"\nError running commodity metrics: {e}")

        else:
            print("\nInvalid choice. Please select 1-16.")


if __name__ == "__main__":
    interactive_menu()
