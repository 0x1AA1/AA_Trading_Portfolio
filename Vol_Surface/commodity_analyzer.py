"""
Commodity-Specific Analysis Module

Provides commodity futures term structure analysis for ETF options trading.

Features:
- Contango/backwardation measurement
- Roll yield calculation
- Seasonality pattern detection
- Term structure slope analysis
- Convenience yield estimation

Supports commodity ETFs: USO, UNG, GLD, SLV, CORN, WEAT, etc.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommodityAnalyzer:
    """
    Analyzes commodity-specific metrics for options trading edge
    """

    # Commodity futures mappings
    COMMODITY_FUTURES_MAP = {
        'USO': 'CL=F',      # Crude Oil
        'UNG': 'NG=F',      # Natural Gas
        'GLD': 'GC=F',      # Gold
        'SLV': 'SI=F',      # Silver
        'PPLT': 'PL=F',     # Platinum
        'PALL': 'PA=F',     # Palladium
        'CORN': 'ZC=F',     # Corn
        'WEAT': 'ZW=F',     # Wheat
        'SOYB': 'ZS=F',     # Soybeans
        'UGA': 'RB=F',      # Gasoline
        'DBA': 'ZC=F',      # Agriculture (use corn as proxy)
        'DBC': 'CL=F',      # Commodities (use crude as proxy)
        'XLE': 'CL=F',      # Energy sector (use crude as proxy)
    }

    # Seasonality profiles (historical patterns)
    SEASONALITY_PATTERNS = {
        'NG=F': {  # Natural Gas
            'peak_months': [11, 12, 1, 2],  # Winter heating demand
            'trough_months': [5, 6, 7, 8],  # Spring/summer lull
            'description': 'Winter heating demand drives seasonal pattern'
        },
        'RB=F': {  # Gasoline
            'peak_months': [5, 6, 7, 8],    # Summer driving season
            'trough_months': [1, 2, 11, 12], # Winter lull
            'description': 'Summer driving season creates demand spike'
        },
        'ZC=F': {  # Corn
            'peak_months': [6, 7, 8],       # Weather risk during pollination
            'trough_months': [11, 12, 1],   # Post-harvest certainty
            'description': 'Growing season weather uncertainty'
        },
        'ZW=F': {  # Wheat
            'peak_months': [5, 6, 7],       # Spring wheat planting
            'trough_months': [9, 10, 11],   # Post-harvest
            'description': 'Planting and growing season risk'
        },
        'ZS=F': {  # Soybeans
            'peak_months': [6, 7, 8],       # Weather-sensitive growth period
            'trough_months': [10, 11, 12],  # Post-harvest
            'description': 'Critical summer growing period'
        }
    }

    def __init__(self):
        """Initialize commodity analyzer"""
        logger.info("Commodity Analyzer initialized")

    def analyze_commodity(
        self,
        ticker: str,
        current_price: float,
        options_df: pd.DataFrame
    ) -> Dict:
        """
        Comprehensive commodity analysis

        Args:
            ticker: Commodity ETF symbol
            current_price: Current ETF price
            options_df: Options chain data

        Returns:
            Dict with commodity metrics
        """
        logger.info(f"Analyzing commodity metrics for {ticker}")

        results = {
            'ticker': ticker,
            'is_commodity': ticker in self.COMMODITY_FUTURES_MAP,
            'futures_symbol': self.COMMODITY_FUTURES_MAP.get(ticker, None)
        }

        if not results['is_commodity']:
            logger.info(f"{ticker} not recognized as commodity ETF")
            return results

        # Term structure analysis
        term_structure = self._analyze_term_structure(ticker, options_df)
        results['term_structure'] = term_structure

        # Roll yield estimation
        roll_yield = self._estimate_roll_yield(term_structure)
        results['roll_yield'] = roll_yield

        # Seasonality analysis
        seasonality = self._analyze_seasonality(ticker)
        results['seasonality'] = seasonality

        # Seasonality-aware IV interpretation (cyclicality)
        if term_structure.get('status') == 'success' and seasonality.get('status') == 'success':
            phase = seasonality.get('current_phase')
            front_iv = term_structure.get('front_month_iv')
            bias = 'neutral'
            comment = 'No strong seasonal IV bias detected.'
            if phase == 'PEAK_SEASON':
                bias = 'elevated_seasonal'
                comment = (
                    'Front-month IV is likely seasonally elevated due to cyclical factors '
                    f'(phase: {phase}). Interpret high IV as seasonally normal rather than acute stress.'
                )
            elif phase == 'TROUGH_SEASON':
                bias = 'depressed_seasonal'
                comment = (
                    'Front-month IV is likely seasonally depressed (phase: TROUGH_SEASON). '
                    'Low IV may reflect cyclicality rather than complacency.'
                )

            results['seasonality_iv_bias'] = bias
            results['seasonality_adjusted_iv_comment'] = comment

        # Trading recommendations
        recommendations = self._generate_recommendations(
            term_structure, roll_yield, seasonality
        )
        results['recommendations'] = recommendations

        logger.info(f"Commodity analysis complete for {ticker}")
        return results

    def _analyze_term_structure(
        self,
        ticker: str,
        options_df: pd.DataFrame
    ) -> Dict:
        """
        Analyze implied volatility term structure

        Returns:
            Dict with term structure metrics
        """
        if options_df.empty or 'impliedVolatility' not in options_df.columns:
            return {'status': 'insufficient_data'}

        # Group by expiration
        options_df = options_df.copy()
        options_df['expiration'] = pd.to_datetime(options_df['expiration'])

        # Calculate average IV per expiration (ATM options)
        # ATM defined as strike within 5% of spot
        term_structure_data = []

        expirations = sorted(options_df['expiration'].unique())

        for exp_date in expirations:
            exp_options = options_df[options_df['expiration'] == exp_date]

            if exp_options.empty:
                continue

            # Get ATM-ish options (closest to current price)
            # Work on a copy to avoid chained assignment warnings
            exp_options = exp_options.copy()
            exp_options['moneyness'] = abs(exp_options['strike'] / exp_options['strike'].median() - 1)
            atm_options = exp_options[exp_options['moneyness'] < 0.05]

            if atm_options.empty:
                atm_options = exp_options.nsmallest(5, 'moneyness')

            avg_iv = atm_options['impliedVolatility'].mean()
            dte = (exp_date - pd.Timestamp.now()).days

            term_structure_data.append({
                'expiration': exp_date,
                'days_to_expiry': dte,
                'avg_iv': avg_iv
            })

        if len(term_structure_data) < 2:
            return {'status': 'insufficient_expirations'}

        # Calculate slope
        term_df = pd.DataFrame(term_structure_data)
        term_df = term_df.sort_values('days_to_expiry')

        # Front month vs back month slope
        front_iv = term_df.iloc[0]['avg_iv']
        back_iv = term_df.iloc[-1]['avg_iv']
        slope = (back_iv - front_iv) / (term_df.iloc[-1]['days_to_expiry'] - term_df.iloc[0]['days_to_expiry'])

        # Classify term structure
        if front_iv > back_iv * 1.10:
            structure_type = 'STEEP_BACKWARDATION'
            description = 'Front month IV significantly higher - near-term stress'
        elif front_iv > back_iv * 1.03:
            structure_type = 'BACKWARDATION'
            description = 'Front month IV higher - short-term uncertainty'
        elif back_iv > front_iv * 1.10:
            structure_type = 'STEEP_CONTANGO'
            description = 'Back month IV significantly higher - long-term uncertainty'
        elif back_iv > front_iv * 1.03:
            structure_type = 'CONTANGO'
            description = 'Back month IV higher - normal market'
        else:
            structure_type = 'FLAT'
            description = 'Relatively flat term structure'

        return {
            'status': 'success',
            'structure_type': structure_type,
            'description': description,
            'front_month_iv': front_iv,
            'back_month_iv': back_iv,
            'slope': slope,
            'slope_annualized': slope * 365,
            'term_data': term_df.to_dict('records')
        }

    def _estimate_roll_yield(self, term_structure: Dict) -> Dict:
        """
        Estimate roll yield impact on calendar spreads

        Roll yield is the P&L from holding and rolling futures positions.
        In contango: negative roll yield (buy high, sell low)
        In backwardation: positive roll yield (buy low, sell high)
        """
        if term_structure.get('status') != 'success':
            return {'status': 'no_term_structure'}

        structure_type = term_structure['structure_type']
        slope = term_structure.get('slope_annualized', 0)

        # Estimate monthly roll cost
        # Approximation: slope * (30/365) gives monthly impact
        monthly_roll_pct = slope * (30/365)

        if 'CONTANGO' in structure_type:
            # Negative roll yield - cost to roll forward
            roll_yield_annual = -abs(slope) * 100  # Convert to percentage
            impact = 'NEGATIVE'
            description = f'Contango implies negative roll yield ~{abs(roll_yield_annual):.1f}% annually'
        elif 'BACKWARDATION' in structure_type:
            # Positive roll yield - profit from roll
            roll_yield_annual = abs(slope) * 100
            impact = 'POSITIVE'
            description = f'Backwardation implies positive roll yield ~{roll_yield_annual:.1f}% annually'
        else:
            roll_yield_annual = 0
            impact = 'NEUTRAL'
            description = 'Flat structure implies minimal roll yield'

        return {
            'status': 'success',
            'impact': impact,
            'roll_yield_annual_pct': roll_yield_annual,
            'roll_yield_monthly_pct': monthly_roll_pct * 100,
            'description': description,
            'trading_implication': self._roll_yield_trading_implication(impact, structure_type)
        }

    def _roll_yield_trading_implication(self, impact: str, structure_type: str) -> str:
        """Generate trading implications from roll yield"""
        if impact == 'NEGATIVE':
            return (
                'Calendar spreads favor SELLING short-dated / BUYING long-dated. '
                'ETF will lose value over time due to roll costs. '
                'Consider bearish options strategies or avoid long ETF positions.'
            )
        elif impact == 'POSITIVE':
            return (
                'Calendar spreads favor BUYING short-dated / SELLING long-dated. '
                'ETF benefits from positive roll. '
                'Consider bullish strategies or long ETF positions.'
            )
        else:
            return 'Neutral roll yield - focus on other factors'

    def _analyze_seasonality(self, ticker: str) -> Dict:
        """
        Analyze seasonal patterns for commodity

        Returns:
            Dict with seasonality info
        """
        futures_symbol = self.COMMODITY_FUTURES_MAP.get(ticker)
        pattern = self.SEASONALITY_PATTERNS.get(futures_symbol)

        if pattern is None:
            return {
                'status': 'no_pattern',
                'description': f'No seasonal pattern defined for {ticker}'
            }

        current_month = datetime.now().month

        # Check if we're in peak or trough season
        in_peak = current_month in pattern['peak_months']
        in_trough = current_month in pattern['trough_months']

        if in_peak:
            phase = 'PEAK_SEASON'
            description = f'Currently in peak volatility season. {pattern["description"]}'
        elif in_trough:
            phase = 'TROUGH_SEASON'
            description = f'Currently in low volatility season. {pattern["description"]}'
        else:
            phase = 'TRANSITION'
            description = f'Transitioning between seasons. {pattern["description"]}'

        # Calculate months until next peak
        peak_months = pattern['peak_months']
        months_until_peak = min(
            (pm - current_month) % 12 for pm in peak_months
        )

        return {
            'status': 'success',
            'current_phase': phase,
            'description': description,
            'peak_months': pattern['peak_months'],
            'trough_months': pattern['trough_months'],
            'months_until_peak': months_until_peak,
            'trading_implication': self._seasonality_trading_implication(phase, months_until_peak)
        }

    def _seasonality_trading_implication(self, phase: str, months_until_peak: int) -> str:
        """Generate trading implications from seasonality"""
        if phase == 'PEAK_SEASON':
            return (
                'Currently in high volatility season. '
                'IV likely elevated - consider SELLING volatility if IV > HV. '
                'Wait for IV spike to sell straddles/strangles.'
            )
        elif phase == 'TROUGH_SEASON':
            return (
                'Currently in low volatility season. '
                'IV likely compressed - consider BUYING volatility if cheap. '
                f'Build positions ahead of peak season ({months_until_peak} months away).'
            )
        else:
            return (
                f'Transitioning to peak season in {months_until_peak} months. '
                'Consider building long volatility positions if IV still low. '
                'Monitor for seasonal IV increase.'
            )

    def _generate_recommendations(
        self,
        term_structure: Dict,
        roll_yield: Dict,
        seasonality: Dict
    ) -> List[str]:
        """
        Generate actionable trading recommendations

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Term structure recommendations
        if term_structure.get('status') == 'success':
            structure_type = term_structure['structure_type']

            if 'STEEP_BACKWARDATION' in structure_type:
                recommendations.append(
                    'TACTICAL: Term structure in steep backwardation. '
                    'Consider selling short-term vol (straddles/strangles) '
                    'and buying longer-dated protection. '
                    'Near-term stress likely to resolve.'
                )
            elif 'STEEP_CONTANGO' in structure_type:
                recommendations.append(
                    'TACTICAL: Term structure in steep contango. '
                    'Long-term uncertainty elevated. '
                    'Consider buying back-month volatility or calendar spreads '
                    '(sell front, buy back).'
                )

        # Roll yield recommendations
        if roll_yield.get('status') == 'success':
            impact = roll_yield['impact']

            if impact == 'NEGATIVE':
                recommendations.append(
                    f'STRUCTURAL: Negative roll yield ({roll_yield["roll_yield_annual_pct"]:.1f}%/year). '
                    'ETF will erode over time. Favor bearish strategies or short-term trades only. '
                    'Avoid buy-and-hold on underlying ETF.'
                )
            elif impact == 'POSITIVE':
                recommendations.append(
                    f'STRUCTURAL: Positive roll yield (+{roll_yield["roll_yield_annual_pct"]:.1f}%/year). '
                    'ETF benefits from term structure. Consider bullish strategies or long ETF exposure.'
                )

        # Seasonality recommendations
        if seasonality.get('status') == 'success':
            phase = seasonality['current_phase']
            months_until_peak = seasonality['months_until_peak']

            if phase == 'TROUGH_SEASON' and months_until_peak <= 2:
                recommendations.append(
                    f'SEASONAL: Approaching peak volatility season in {months_until_peak} months. '
                    'Build long volatility positions now while IV is seasonally low. '
                    'Target expirations covering peak season.'
                )
            elif phase == 'PEAK_SEASON':
                recommendations.append(
                    'SEASONAL: Currently in peak volatility season. '
                    'IV likely elevated. Look for opportunities to sell expensive vol. '
                    'Consider short straddles/strangles if IV percentile > 70.'
                )

        # Combined strategy recommendation
        if len(recommendations) >= 2:
            recommendations.append(
                'STRATEGY: Combine structural (roll yield) and seasonal factors. '
                'Use term structure for timing entries. '
                'Size positions based on confidence in all three factors aligning.'
            )

        if not recommendations:
            recommendations.append(
                'No strong directional signals from commodity-specific factors. '
                'Focus on standard IV-HV spread analysis.'
            )

        return recommendations


if __name__ == "__main__":
    # Test commodity analyzer
    print("Testing Commodity Analyzer\n")

    analyzer = CommodityAnalyzer()

    # Test with sample data
    test_tickers = ['USO', 'UNG', 'GLD', 'AAPL']

    for ticker in test_tickers:
        print(f"\n{'='*60}")
        print(f"Analyzing {ticker}")
        print('='*60)

        # Mock options data
        mock_options = pd.DataFrame({
            'strike': [70, 72, 75, 78, 80],
            'expiration': [
                pd.Timestamp.now() + timedelta(days=30),
                pd.Timestamp.now() + timedelta(days=30),
                pd.Timestamp.now() + timedelta(days=60),
                pd.Timestamp.now() + timedelta(days=60),
                pd.Timestamp.now() + timedelta(days=90)
            ],
            'impliedVolatility': [0.30, 0.31, 0.28, 0.29, 0.26]
        })

        result = analyzer.analyze_commodity(ticker, 75.0, mock_options)

        print(f"\nIs Commodity: {result['is_commodity']}")

        if result['is_commodity']:
            print(f"Futures Symbol: {result['futures_symbol']}")

            if result['term_structure'].get('status') == 'success':
                ts = result['term_structure']
                print(f"\nTerm Structure: {ts['structure_type']}")
                print(f"Description: {ts['description']}")
                print(f"Front Month IV: {ts['front_month_iv']:.2%}")
                print(f"Back Month IV: {ts['back_month_iv']:.2%}")

            if result['roll_yield'].get('status') == 'success':
                ry = result['roll_yield']
                print(f"\nRoll Yield Impact: {ry['impact']}")
                print(f"Annual Roll Yield: {ry['roll_yield_annual_pct']:+.2f}%")
                print(f"Trading Implication: {ry['trading_implication']}")

            if result['seasonality'].get('status') == 'success':
                seas = result['seasonality']
                print(f"\nSeasonality Phase: {seas['current_phase']}")
                print(f"Description: {seas['description']}")
                print(f"Trading Implication: {seas['trading_implication']}")

            print(f"\nRECOMMENDATIONS:")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"{i}. {rec}")

    print("\nCommodity Analyzer test complete!")
