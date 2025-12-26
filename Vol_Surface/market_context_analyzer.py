"""
Market Context Analyzer

Provides fundamental and technical context for options trading decisions:
- Recent price momentum and trend analysis
- Analyst ratings and price targets
- 52-week high/low positioning
- Related asset correlation analysis
- Volume and liquidity metrics

Author: Aoures ABDI
Version: 1.0
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class MarketContextAnalyzer:
    """
    Analyzes broader market context beyond options Greeks and volatility
    """

    # Commodity/sector relationships for automatic detection
    ASSET_RELATIONSHIPS = {
        'gold_miners': {
            'tickers': ['NEM', 'GOLD', 'AEM', 'ABX', 'KGC', 'AU', 'FNV', 'WPM', 'RGLD'],
            'related': 'GC=F',
            'name': 'Gold Futures'
        },
        'oil_gas': {
            'tickers': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'HAL', 'OXY'],
            'related': 'CL=F',
            'name': 'Crude Oil Futures'
        },
        'tech': {
            'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC'],
            'related': 'QQQ',
            'name': 'Nasdaq-100 ETF'
        },
        'financials': {
            'tickers': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW'],
            'related': '^TNX',
            'name': '10-Year Treasury Yield'
        },
        'semiconductors': {
            'tickers': ['NVDA', 'AMD', 'INTC', 'AVGO', 'TSM', 'QCOM', 'MU', 'AMAT'],
            'related': 'SOXX',
            'name': 'Semiconductor ETF'
        }
    }

    def __init__(self):
        """Initialize the market context analyzer"""
        pass

    def analyze_comprehensive_context(
        self,
        ticker: str,
        lookback_days: int = 90
    ) -> Dict:
        """
        Get complete market context in one call

        Args:
            ticker: Stock symbol
            lookback_days: Historical period to analyze

        Returns:
            Comprehensive context dictionary
        """
        logger.info(f"Analyzing comprehensive market context for {ticker}")

        context = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'price_context': self.fetch_price_context(ticker, lookback_days),
            'analyst_context': self.fetch_analyst_consensus(ticker),
            'related_asset_context': self.fetch_related_asset_context(ticker)
        }

        # Add derived insights
        context['market_positioning'] = self._assess_market_positioning(context)

        return context

    def fetch_price_context(self, ticker: str, lookback_days: int = 90) -> Dict:
        """
        Comprehensive price history and momentum analysis

        Args:
            ticker: Stock symbol
            lookback_days: Days of history to analyze

        Returns:
            Dictionary with price metrics
        """
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            hist = stock.history(period=f'{lookback_days}d')

            if hist.empty:
                logger.warning(f"No price history for {ticker}")
                return {}

            current = hist['Close'].iloc[-1]

            # Calculate multi-period returns
            returns = self._calculate_period_returns(hist, current)

            # 52-week high/low analysis
            week52_high = hist['Close'].max()
            week52_low = hist['Close'].min()
            distance_from_high = (current / week52_high - 1)
            distance_from_low = (current / week52_low - 1)
            percentile = (current - week52_low) / (week52_high - week52_low) if week52_high > week52_low else 0.5

            # Volatility metrics
            daily_returns = hist['Close'].pct_change().dropna()
            realized_vol = daily_returns.std() * np.sqrt(252)

            # Volume analysis
            avg_volume = hist['Volume'].mean()
            recent_volume = hist['Volume'].tail(5).mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

            # Trend determination
            trend = self._determine_trend(hist)

            # Momentum score
            momentum_score = self._calculate_momentum_score(returns)

            return {
                'current_price': float(current),
                '1w_return': returns.get('1w', 0),
                '1m_return': returns.get('1m', 0),
                '3m_return': returns.get('3m', 0),
                'ytd_return': returns.get('ytd', 0),

                '52w_high': float(week52_high),
                '52w_low': float(week52_low),
                'distance_from_52w_high': float(distance_from_high),
                'distance_from_52w_low': float(distance_from_low),
                'percentile_in_52w_range': float(percentile),

                'realized_volatility_90d': float(realized_vol),
                'avg_daily_volume': float(avg_volume),
                'recent_volume_ratio': float(volume_ratio),

                'price_trend': trend,
                'momentum_score': momentum_score
            }

        except Exception as e:
            logger.error(f"Error fetching price context for {ticker}: {e}")
            return {}

    def fetch_analyst_consensus(self, ticker: str) -> Dict:
        """
        Get analyst ratings, price targets, and consensus

        Args:
            ticker: Stock symbol

        Returns:
            Analyst data dictionary
        """
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            info = stock.info

            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            target_mean = info.get('targetMeanPrice')
            target_high = info.get('targetHighPrice')
            target_low = info.get('targetLowPrice')

            # Calculate upside/downside
            upside_to_mean = None
            upside_to_high = None
            downside_to_low = None

            if current_price and current_price > 0:
                if target_mean:
                    upside_to_mean = (target_mean / current_price - 1)
                if target_high:
                    upside_to_high = (target_high / current_price - 1)
                if target_low:
                    downside_to_low = (target_low / current_price - 1)

            return {
                'current_price': float(current_price) if current_price else None,
                'target_mean': float(target_mean) if target_mean else None,
                'target_high': float(target_high) if target_high else None,
                'target_low': float(target_low) if target_low else None,
                'num_analysts': info.get('numberOfAnalystOpinions'),
                'recommendation_key': info.get('recommendationKey'),
                'recommendation_mean': info.get('recommendationMean'),

                'upside_to_mean': float(upside_to_mean) if upside_to_mean is not None else None,
                'upside_to_high': float(upside_to_high) if upside_to_high is not None else None,
                'downside_to_low': float(downside_to_low) if downside_to_low is not None else None,

                'analyst_support_level': target_low,
                'analyst_resistance_level': target_high
            }

        except Exception as e:
            logger.error(f"Error fetching analyst data for {ticker}: {e}")
            return {}

    def fetch_related_asset_context(
        self,
        ticker: str,
        related_ticker: Optional[str] = None
    ) -> Dict:
        """
        Analyze related commodity, index, or sector performance

        Args:
            ticker: Stock symbol
            related_ticker: Specific related asset (auto-detects if None)

        Returns:
            Related asset context
        """
        if not related_ticker:
            related_ticker, asset_name = self._auto_detect_related_asset(ticker)
        else:
            asset_name = related_ticker

        if not related_ticker:
            return {'related_asset': None, 'correlation_available': False}

        try:
            import yfinance as yf

            related = yf.Ticker(related_ticker)
            hist = related.history(period='3mo')

            if hist.empty:
                return {'related_asset': related_ticker, 'data_available': False}

            current = hist['Close'].iloc[-1]
            month_ago = hist['Close'].iloc[-21] if len(hist) >= 21 else hist['Close'].iloc[0]
            three_mo_ago = hist['Close'].iloc[0]

            range_3m_high = hist['Close'].max()
            range_3m_low = hist['Close'].min()
            percentile = (current - range_3m_low) / (range_3m_high - range_3m_low) if range_3m_high > range_3m_low else 0.5

            return {
                'related_asset': related_ticker,
                'asset_name': asset_name,
                'current_level': float(current),
                '1m_return': float((current / month_ago - 1)) if month_ago > 0 else 0,
                '3m_return': float((current / three_mo_ago - 1)) if three_mo_ago > 0 else 0,
                '3m_high': float(range_3m_high),
                '3m_low': float(range_3m_low),
                'percentile_in_3m_range': float(percentile),
                'correlation_implication': self._interpret_correlation(
                    ticker, related_ticker, percentile
                )
            }

        except Exception as e:
            logger.error(f"Error fetching related asset {related_ticker}: {e}")
            return {'related_asset': related_ticker, 'error': str(e)}

    def _calculate_period_returns(self, hist: pd.DataFrame, current_price: float) -> Dict:
        """Calculate returns over various periods"""
        returns = {}

        periods = {
            '1w': 5,
            '1m': 21,
            '3m': 63
        }

        for period_name, days in periods.items():
            if len(hist) >= days:
                past_price = hist['Close'].iloc[-days]
                if past_price > 0:
                    returns[period_name] = float(current_price / past_price - 1)

        # YTD return
        try:
            year_start = pd.Timestamp(datetime.now().year, 1, 1)
            ytd_data = hist[hist.index >= year_start]
            if not ytd_data.empty and ytd_data['Close'].iloc[0] > 0:
                returns['ytd'] = float(current_price / ytd_data['Close'].iloc[0] - 1)
        except:
            pass

        return returns

    def _determine_trend(self, hist: pd.DataFrame) -> str:
        """Determine price trend using moving averages"""
        try:
            sma_20 = hist['Close'].tail(20).mean()
            sma_50 = hist['Close'].tail(50).mean() if len(hist) >= 50 else sma_20
            current = hist['Close'].iloc[-1]

            if current > sma_20 > sma_50:
                return 'STRONG_UPTREND'
            elif current > sma_20:
                return 'UPTREND'
            elif current < sma_20 < sma_50:
                return 'STRONG_DOWNTREND'
            elif current < sma_20:
                return 'DOWNTREND'
            else:
                return 'SIDEWAYS'
        except:
            return 'UNKNOWN'

    def _calculate_momentum_score(self, returns: Dict) -> int:
        """
        Calculate momentum score 0-100
        Higher = stronger upward momentum
        """
        score = 50  # Neutral baseline

        # Short-term (1 week): weight 20
        if '1w' in returns:
            score += min(20, max(-20, returns['1w'] * 100))

        # Medium-term (1 month): weight 15
        if '1m' in returns:
            score += min(15, max(-15, returns['1m'] * 50))

        # Longer-term (3 months): weight 15
        if '3m' in returns:
            score += min(15, max(-15, returns['3m'] * 33))

        return int(max(0, min(100, score)))

    def _auto_detect_related_asset(self, ticker: str) -> tuple:
        """
        Auto-detect related commodity/index for a stock

        Returns:
            (related_ticker, asset_name) tuple
        """
        ticker_upper = ticker.upper()

        for sector, info in self.ASSET_RELATIONSHIPS.items():
            if ticker_upper in info['tickers']:
                return (info['related'], info['name'])

        return (None, None)

    def _interpret_correlation(
        self,
        ticker: str,
        related_ticker: str,
        percentile: float
    ) -> str:
        """Generate human-readable correlation interpretation"""

        if 'GC=F' in related_ticker:  # Gold
            if percentile > 0.8:
                return "Gold near 3-month highs - bullish tailwind for gold miners"
            elif percentile < 0.2:
                return "Gold near 3-month lows - headwind for gold miners"
            else:
                return "Gold in mid-range - neutral for miners"

        elif 'CL=F' in related_ticker:  # Oil
            if percentile > 0.8:
                return "Oil prices elevated - supportive for energy stocks"
            elif percentile < 0.2:
                return "Oil prices depressed - headwind for energy stocks"
            else:
                return "Oil in mid-range - neutral impact"

        elif 'QQQ' in related_ticker:  # Tech/Nasdaq
            if percentile > 0.8:
                return "Nasdaq near highs - tech sector strength"
            elif percentile < 0.2:
                return "Nasdaq weakness - tech sector pressure"
            else:
                return "Nasdaq mid-range - sector neutral"

        elif '^TNX' in related_ticker:  # Interest rates
            if percentile > 0.8:
                return "Rates elevated - potential headwind for financials"
            elif percentile < 0.2:
                return "Rates low - reduced net interest margin pressure"
            else:
                return "Rates mid-range - neutral for financials"

        return "Related asset context available"

    def _assess_market_positioning(self, context: Dict) -> Dict:
        """
        Synthesize all context into market positioning assessment

        Returns:
            Assessment dictionary with positioning metrics
        """
        price_ctx = context.get('price_context', {})
        analyst_ctx = context.get('analyst_context', {})
        related_ctx = context.get('related_asset_context', {})

        assessment = {
            'overall_positioning': 'NEUTRAL',
            'risk_flags': [],
            'opportunity_flags': []
        }

        # Check if stock is extended
        percentile = price_ctx.get('percentile_in_52w_range', 0.5)
        if percentile > 0.90:
            assessment['risk_flags'].append('Stock in top 10% of 52-week range - extended')
            assessment['overall_positioning'] = 'EXTENDED'
        elif percentile < 0.10:
            assessment['opportunity_flags'].append('Stock in bottom 10% of 52-week range - potential value')
            assessment['overall_positioning'] = 'OVERSOLD'

        # Check momentum
        momentum = price_ctx.get('momentum_score', 50)
        if momentum > 75:
            assessment['risk_flags'].append(f'High momentum ({momentum}/100) - potential exhaustion')
        elif momentum < 25:
            assessment['opportunity_flags'].append(f'Low momentum ({momentum}/100) - potential reversal')

        # Check analyst sentiment
        upside_to_mean = analyst_ctx.get('upside_to_mean')
        if upside_to_mean is not None:
            if upside_to_mean < -0.05:
                assessment['risk_flags'].append(f'Trading above analyst mean target ({upside_to_mean*100:.1f}% upside)')
            elif upside_to_mean > 0.20:
                assessment['opportunity_flags'].append(f'Significant upside to analyst targets (+{upside_to_mean*100:.1f}%)')

        # Check related asset
        related_percentile = related_ctx.get('percentile_in_3m_range')
        if related_percentile is not None:
            if related_percentile > 0.90:
                assessment['risk_flags'].append('Related asset near 3-month highs - limited tailwind')
            elif related_percentile < 0.10:
                assessment['opportunity_flags'].append('Related asset near lows - potential support if recovers')

        return assessment
