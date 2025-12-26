"""
Strategy Backtester

Backtests option strategies with historical data to validate performance.

Features:
- Historical volatility surface reconstruction
- Strategy P&L tracking over time
- Performance metrics (Sharpe, max drawdown, win rate)
- Transaction costs and slippage modeling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyBacktester:
    """
    Backtests option strategies using historical data
    """

    def __init__(
        self,
        commission_per_contract: float = 0.65,
        slippage_bps: float = 5.0
    ):
        """
        Initialize backtester

        Args:
            commission_per_contract: Commission per option contract
            slippage_bps: Slippage in basis points (0.05 = 5 bps)
        """
        self.commission = commission_per_contract
        self.slippage_bps = slippage_bps / 10000  # Convert to decimal

        logger.info(f"Strategy Backtester initialized (commission: ${commission_per_contract}, slippage: {slippage_bps} bps)")

    def backtest_strategy(
        self,
        ticker: str,
        strategy_type: str,
        entry_rules: Dict,
        exit_rules: Dict,
        start_date: str,
        end_date: str,
        initial_capital: float = 10000.0
    ) -> Dict:
        """
        Backtest a specific strategy type over historical period

        Args:
            ticker: Underlying ticker
            strategy_type: 'STRADDLE', 'STRANGLE', 'CALENDAR', 'BUTTERFLY'
            entry_rules: Entry criteria dict
            exit_rules: Exit criteria dict
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital

        Returns:
            Dict with backtest results and performance metrics
        """
        logger.info(f"Backtesting {strategy_type} on {ticker} from {start_date} to {end_date}")

        # Fetch historical data
        historical_data = self._fetch_historical_data(ticker, start_date, end_date)

        if historical_data.empty:
            logger.warning("No historical data available")
            return self._empty_results()

        # Simulate strategy trades
        trades = self._simulate_trades(
            historical_data,
            strategy_type,
            entry_rules,
            exit_rules
        )

        if not trades:
            logger.warning("No trades executed during backtest period")
            return self._empty_results()

        # Calculate performance metrics
        performance = self._calculate_performance(trades, initial_capital)

        # Generate equity curve
        equity_curve = self._generate_equity_curve(trades, initial_capital)

        results = {
            'ticker': ticker,
            'strategy_type': strategy_type,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'trades': trades,
            'performance': performance,
            'equity_curve': equity_curve
        }

        logger.info(f"Backtest complete: {len(trades)} trades, {performance['total_return']:.2f}% return")

        return results

    def _fetch_historical_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical price and volatility data

        Note: Simplified version using yfinance. Full implementation would
        require historical options data from professional data provider.
        """
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)

            if hist.empty:
                return pd.DataFrame()

            # Calculate rolling volatility
            hist['returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
            hist['volatility_30d'] = hist['returns'].rolling(30).std() * np.sqrt(252)

            # Drop NaN
            hist = hist.dropna()

            logger.info(f"Fetched {len(hist)} days of historical data for {ticker}")

            return hist

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def _simulate_trades(
        self,
        historical_data: pd.DataFrame,
        strategy_type: str,
        entry_rules: Dict,
        exit_rules: Dict
    ) -> List[Dict]:
        """
        Simulate strategy trades based on entry/exit rules

        Simplified simulation using historical price and volatility.
        Real implementation would require full options chain history.
        """
        trades = []
        position = None  # Current open position

        for idx, row in historical_data.iterrows():
            date = idx
            price = row['Close']
            vol = row.get('volatility_30d', 0.30)

            # Check if we should enter a new position
            if position is None:
                should_enter = self._check_entry_rules(
                    price, vol, entry_rules, strategy_type
                )

                if should_enter:
                    position = self._enter_position(
                        date, price, vol, strategy_type
                    )

            # Check if we should exit existing position
            else:
                should_exit = self._check_exit_rules(
                    position, date, price, vol, exit_rules
                )

                if should_exit:
                    trade = self._exit_position(position, date, price, vol)
                    trades.append(trade)
                    position = None

        # Close any remaining position at end
        if position is not None:
            final_date = historical_data.index[-1]
            final_price = historical_data.iloc[-1]['Close']
            final_vol = historical_data.iloc[-1].get('volatility_30d', 0.30)

            trade = self._exit_position(position, final_date, final_price, final_vol)
            trades.append(trade)

        return trades

    def _check_entry_rules(
        self,
        price: float,
        vol: float,
        entry_rules: Dict,
        strategy_type: str
    ) -> bool:
        """
        Check if entry conditions are met

        Example entry_rules:
        {
            'volatility_threshold': 0.25,  # Enter when IV > 25%
            'volatility_percentile': 75,   # Enter when IV > 75th percentile
            'price_trend': 'neutral'       # 'up', 'down', 'neutral'
        }
        """
        # Simplified entry logic
        vol_threshold = entry_rules.get('volatility_threshold', 0.20)

        if vol > vol_threshold:
            return True

        return False

    def _check_exit_rules(
        self,
        position: Dict,
        current_date: datetime,
        price: float,
        vol: float,
        exit_rules: Dict
    ) -> bool:
        """
        Check if exit conditions are met

        Example exit_rules:
        {
            'profit_target': 50,     # Exit at 50% profit
            'stop_loss': -30,        # Exit at -30% loss
            'days_to_expiry': 7,     # Exit 7 days before expiry
            'volatility_drop': 0.10  # Exit if vol drops 10%
        }
        """
        entry_date = position['entry_date']
        entry_price = position['entry_price']
        expiry_date = position['expiry_date']

        # Days to expiry
        days_left = (expiry_date - current_date).days
        if days_left <= exit_rules.get('days_to_expiry', 7):
            return True

        # Profit/loss thresholds
        current_value = self._estimate_position_value(position, price, vol)
        pnl_pct = ((current_value - entry_price) / entry_price) * 100

        profit_target = exit_rules.get('profit_target', 100)
        stop_loss = exit_rules.get('stop_loss', -50)

        if pnl_pct >= profit_target or pnl_pct <= stop_loss:
            return True

        return False

    def _enter_position(
        self,
        date: datetime,
        price: float,
        vol: float,
        strategy_type: str
    ) -> Dict:
        """
        Enter a new position

        Simplified: Uses Black-Scholes approximation for option pricing
        """
        # Default: 30-day to expiry
        expiry_date = date + timedelta(days=30)

        # Estimate option prices
        T = 30 / 365

        if strategy_type == 'STRADDLE':
            # ATM straddle
            strike = price

            # Simplified BS pricing
            call_price = price * vol * np.sqrt(T) * 0.40  # Approximation
            put_price = call_price

            entry_cost = call_price + put_price

        elif strategy_type == 'STRANGLE':
            # 5% OTM strangle
            call_strike = price * 1.05
            put_strike = price * 0.95

            call_price = price * vol * np.sqrt(T) * 0.25
            put_price = call_price

            entry_cost = call_price + put_price
            strike = price  # Reference

        elif strategy_type == 'CALENDAR':
            # Calendar spread approximation
            strike = price
            entry_cost = price * vol * np.sqrt(T) * 0.10

        else:
            # Default
            strike = price
            entry_cost = price * vol * np.sqrt(T) * 0.30

        # Apply transaction costs
        entry_cost += self.commission * 2  # 2 legs
        entry_cost *= (1 + self.slippage_bps)

        position = {
            'entry_date': date,
            'entry_price': entry_cost,
            'underlying_price': price,
            'volatility': vol,
            'strike': strike,
            'expiry_date': expiry_date,
            'strategy_type': strategy_type
        }

        return position

    def _exit_position(
        self,
        position: Dict,
        date: datetime,
        price: float,
        vol: float
    ) -> Dict:
        """
        Exit existing position and record trade
        """
        entry_cost = position['entry_price']

        # Estimate exit value
        exit_value = self._estimate_position_value(position, price, vol)

        # Apply transaction costs
        exit_value -= self.commission * 2
        exit_value *= (1 - self.slippage_bps)

        # Calculate P&L
        pnl = exit_value - entry_cost
        pnl_pct = (pnl / entry_cost) * 100

        # Days held
        days_held = (date - position['entry_date']).days

        trade = {
            'entry_date': position['entry_date'],
            'exit_date': date,
            'days_held': days_held,
            'strategy_type': position['strategy_type'],
            'entry_price': entry_cost,
            'exit_price': exit_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_underlying': position['underlying_price'],
            'exit_underlying': price,
            'entry_vol': position['volatility'],
            'exit_vol': vol
        }

        return trade

    def _estimate_position_value(
        self,
        position: Dict,
        current_price: float,
        current_vol: float
    ) -> float:
        """
        Estimate current value of position

        Simplified using time decay and volatility change
        """
        strategy_type = position['strategy_type']
        entry_price = position['entry_price']
        entry_vol = position['volatility']

        # Time decay factor
        # Handle timezone-aware datetime from pandas
        expiry = position['expiry_date']
        if hasattr(expiry, 'tz_localize'):
            expiry = expiry.tz_localize(None)
        current_time = pd.Timestamp(datetime.now())
        days_remaining = (expiry - current_time).days
        time_factor = max(days_remaining / 30, 0.1)

        # Volatility factor
        vol_factor = current_vol / entry_vol if entry_vol > 0 else 1.0

        # Simplified value estimation
        if strategy_type in ['STRADDLE', 'STRANGLE']:
            # Long volatility benefits from vol increase
            value = entry_price * time_factor * vol_factor
        else:
            value = entry_price * time_factor

        return max(value, 0)

    def _calculate_performance(
        self,
        trades: List[Dict],
        initial_capital: float
    ) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        if not trades:
            return self._empty_performance()

        # Extract P&Ls
        pnls = [t['pnl'] for t in trades]
        pnl_pcts = [t['pnl_pct'] for t in trades]

        # Win rate
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) if pnls else 0

        # Total return
        total_pnl = sum(pnls)
        total_return = (total_pnl / initial_capital) * 100

        # Average metrics
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_pnl = np.mean(pnls)

        # Sharpe ratio (simplified)
        std_pnl = np.std(pnls) if len(pnls) > 1 else 1
        sharpe = (avg_pnl / std_pnl) if std_pnl > 0 else 0

        # Max drawdown
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        max_drawdown_pct = (max_drawdown / initial_capital) * 100

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        performance = {
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_pnl': avg_pnl,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'profit_factor': profit_factor
        }

        return performance

    def _generate_equity_curve(
        self,
        trades: List[Dict],
        initial_capital: float
    ) -> pd.DataFrame:
        """
        Generate equity curve over time
        """
        equity = [initial_capital]
        dates = [trades[0]['entry_date']]

        for trade in trades:
            equity.append(equity[-1] + trade['pnl'])
            dates.append(trade['exit_date'])

        equity_curve = pd.DataFrame({
            'date': dates,
            'equity': equity
        })

        return equity_curve

    def _empty_results(self) -> Dict:
        """Return empty results structure"""
        return {
            'ticker': '',
            'strategy_type': '',
            'trades': [],
            'performance': self._empty_performance(),
            'equity_curve': pd.DataFrame()
        }

    def _empty_performance(self) -> Dict:
        """Return empty performance metrics"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'total_return': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_pnl': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'profit_factor': 0
        }
