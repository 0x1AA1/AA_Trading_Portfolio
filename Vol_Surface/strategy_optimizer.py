"""
Strategy Optimizer

Ranks and optimizes option strategies based on:
1. Expected return / risk metrics
2. Probability of profit
3. Greeks exposure
4. Monte Carlo simulations
5. Alpha signals from screener

Provides best strategy recommendations with detailed analytics
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.stats import norm
import logging

from greeks_calculator import GreeksCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """
    Optimizes and ranks option strategies
    """

    def __init__(self, greeks_calc: GreeksCalculator):
        """
        Initialize optimizer

        Args:
            greeks_calc: GreeksCalculator instance
        """
        self.greeks = greeks_calc

    def rank_strategies(
        self,
        options_df: pd.DataFrame,
        current_price: float,
        historical_vol: float,
        alpha_signals: List[Dict]
    ) -> List[Dict]:
        """
        Evaluate and rank all viable strategies

        Args:
            options_df: Options data
            current_price: Current price
            historical_vol: Historical volatility
            alpha_signals: Signals from alpha screener

        Returns:
            List of strategies ranked by expected value
        """
        logger.info("Ranking strategies...")

        all_strategies = []

        # Build strategies based on alpha signals
        for signal in alpha_signals:
            strategy = self._signal_to_strategy(
                signal,
                options_df,
                current_price,
                historical_vol
            )

            if strategy:
                all_strategies.append(strategy)

        # Sort by risk-adjusted return
        all_strategies = sorted(
            all_strategies,
            key=lambda x: x.get('expected_value', 0) / (x.get('max_risk', 1) + 1),
            reverse=True
        )

        logger.info(f"Ranked {len(all_strategies)} strategies")

        return all_strategies

    def _signal_to_strategy(
        self,
        signal: Dict,
        options_df: pd.DataFrame,
        current_price: float,
        historical_vol: float
    ) -> Optional[Dict]:
        """
        Convert alpha signal to concrete strategy with pricing and analytics
        """
        signal_type = signal.get('type')

        if signal_type == 'IV_HV_SPREAD':
            return self._build_straddle_from_signal(signal, options_df, current_price, historical_vol)

        elif signal_type in ['CHEAP_STRADDLE', 'EXPENSIVE_STRADDLE']:
            return self._build_straddle_from_signal(signal, options_df, current_price, historical_vol)

        elif signal_type == 'STRANGLE_OPPORTUNITY':
            return self._build_strangle_from_signal(signal, options_df, current_price, historical_vol)

        elif signal_type == 'CALENDAR_ARBITRAGE':
            return self._build_calendar_from_signal(signal, options_df, current_price, historical_vol)

        elif signal_type == 'BUTTERFLY_ARBITRAGE':
            return self._build_butterfly_from_signal(signal, options_df, current_price, historical_vol)

        elif signal_type in ['TERM_STRUCTURE', 'VOL_SPIKE']:
            return self._build_from_term_signal(signal, options_df, current_price, historical_vol)

        elif signal_type == 'DIAGONAL_SPREAD':
            return self._build_diagonal_from_signal(signal, options_df, current_price, historical_vol)

        else:
            return None

    def _build_straddle_from_signal(
        self,
        signal: Dict,
        options_df: pd.DataFrame,
        current_price: float,
        historical_vol: float
    ) -> Optional[Dict]:
        """Build straddle strategy from signal"""

        expiry = signal.get('expiration')
        strike = signal.get('strike', current_price)

        # Get call and put
        call = options_df[
            (options_df['strike'] == strike) &
            (options_df['expiration'] == expiry) &
            (options_df['option_type'] == 'call')
        ]

        put = options_df[
            (options_df['strike'] == strike) &
            (options_df['expiration'] == expiry) &
            (options_df['option_type'] == 'put')
        ]

        if call.empty or put.empty:
            return None

        call_price = call.iloc[0].get('mid', call.iloc[0].get('last', 0))
        put_price = put.iloc[0].get('mid', put.iloc[0].get('last', 0))

        if call_price == 0 or put_price == 0:
            return None

        straddle_price = call_price + put_price

        # Determine direction
        is_long = signal.get('signal') in ['BUY_VOLATILITY', 'BUY_STRADDLE']

        # Calculate Greeks
        T = (expiry - pd.Timestamp.now()).days / 365
        iv = signal.get('implied_vol', historical_vol)

        call_greeks = self.greeks.all_greeks(current_price, strike, T, iv, 'call')
        put_greeks = self.greeks.all_greeks(current_price, strike, T, iv, 'put')

        # Straddle Greeks (sum of call + put)
        straddle_greeks = {
            'delta': call_greeks['delta'] + put_greeks['delta'],
            'gamma': call_greeks['gamma'] + put_greeks['gamma'],
            'theta': call_greeks['theta'] + put_greeks['theta'],
            'vega': call_greeks['vega'] + put_greeks['vega']
        }

        # Reverse signs if short
        if not is_long:
            straddle_greeks = {k: -v for k, v in straddle_greeks.items()}

        # Monte Carlo simulation - use IV for pricing, HV for realized moves
        mc_results = self._monte_carlo_straddle(
            current_price, strike, straddle_price, T, historical_vol, iv, is_long
        )

        # Validate results - catch data artifacts
        if (mc_results['prob_profit'] > 0.99 or
            abs(mc_results['expected_return_pct']) > 300 or
            straddle_price < 0.10):
            logger.warning(f"Unrealistic straddle metrics detected - likely data quality issue. Skipping.")
            return None

        # Build strategy dict
        strategy = {
            'type': 'LONG_STRADDLE' if is_long else 'SHORT_STRADDLE',
            'signal_source': signal.get('type'),
            'strike': strike,
            'expiration': expiry,
            'days_to_expiry': int(T * 365),
            'entry_cost': straddle_price if is_long else -straddle_price,
            'max_risk': straddle_price if is_long else np.inf,
            'max_profit': np.inf if is_long else straddle_price,
            'breakeven_up': strike + straddle_price,
            'breakeven_down': strike - straddle_price,
            'implied_vol': iv,
            'historical_vol': historical_vol,
            'iv_hv_spread': iv - historical_vol,
            'greeks': straddle_greeks,
            'probability_profit': mc_results['prob_profit'],
            'expected_value': mc_results['expected_value'],
            'expected_return': mc_results['expected_return_pct'],
            'var_95': mc_results['var_95'],
            'sharpe': mc_results['sharpe'],
            'strength_score': signal.get('strength_score', 50),
            'confidence': signal.get('confidence', 'MEDIUM'),
            'recommendation': signal.get('recommendation', '')
        }

        return strategy

    def _build_strangle_from_signal(
        self,
        signal: Dict,
        options_df: pd.DataFrame,
        current_price: float,
        historical_vol: float
    ) -> Optional[Dict]:
        """Build strangle strategy"""

        expiry = signal.get('expiration')
        put_strike = signal.get('put_strike')
        call_strike = signal.get('call_strike')

        # Get options
        call = options_df[
            (options_df['strike'] == call_strike) &
            (options_df['expiration'] == expiry) &
            (options_df['option_type'] == 'call')
        ]

        put = options_df[
            (options_df['strike'] == put_strike) &
            (options_df['expiration'] == expiry) &
            (options_df['option_type'] == 'put')
        ]

        if call.empty or put.empty:
            return None

        call_price = call.iloc[0].get('mid', call.iloc[0].get('last', 0))
        put_price = put.iloc[0].get('mid', put.iloc[0].get('last', 0))

        if call_price == 0 or put_price == 0:
            return None

        strangle_price = call_price + put_price

        is_long = signal.get('signal') == 'BUY_STRANGLE'

        T = (expiry - pd.Timestamp.now()).days / 365
        iv = signal.get('implied_vol', historical_vol)

        # Greeks
        call_greeks = self.greeks.all_greeks(current_price, call_strike, T, iv, 'call')
        put_greeks = self.greeks.all_greeks(current_price, put_strike, T, iv, 'put')

        strangle_greeks = {
            'delta': call_greeks['delta'] + put_greeks['delta'],
            'gamma': call_greeks['gamma'] + put_greeks['gamma'],
            'theta': call_greeks['theta'] + put_greeks['theta'],
            'vega': call_greeks['vega'] + put_greeks['vega']
        }

        if not is_long:
            strangle_greeks = {k: -v for k, v in strangle_greeks.items()}

        # MC simulation - use IV for pricing, HV for realized moves
        mc_results = self._monte_carlo_strangle(
            current_price, put_strike, call_strike, strangle_price, T, historical_vol, iv, is_long
        )

        # Validate results - catch data artifacts
        if (mc_results['prob_profit'] > 0.99 or
            abs(mc_results['expected_return_pct']) > 300 or
            strangle_price < 0.10):
            logger.warning(f"Unrealistic strangle metrics detected - likely data quality issue. Skipping.")
            return None

        strategy = {
            'type': 'LONG_STRANGLE' if is_long else 'SHORT_STRANGLE',
            'signal_source': signal.get('type'),
            'put_strike': put_strike,
            'call_strike': call_strike,
            'expiration': expiry,
            'days_to_expiry': int(T * 365),
            'entry_cost': strangle_price if is_long else -strangle_price,
            'max_risk': strangle_price if is_long else np.inf,
            'max_profit': np.inf if is_long else strangle_price,
            'breakeven_up': call_strike + strangle_price,
            'breakeven_down': put_strike - strangle_price,
            'implied_vol': iv,
            'historical_vol': historical_vol,
            'greeks': strangle_greeks,
            'probability_profit': mc_results['prob_profit'],
            'expected_value': mc_results['expected_value'],
            'expected_return': mc_results['expected_return_pct'],
            'var_95': mc_results['var_95'],
            'sharpe': mc_results['sharpe'],
            'strength_score': signal.get('strength_score', 50),
            'confidence': signal.get('confidence', 'MEDIUM'),
            'recommendation': signal.get('recommendation', '')
        }

        return strategy

    def _build_diagonal_from_signal(
        self,
        signal: Dict,
        options_df: pd.DataFrame,
        current_price: float,
        historical_vol: float
    ) -> Optional[Dict]:
        """Build a simple diagonal spread from paired expiries and strikes"""
        front = signal.get('front_expiry')
        back = signal.get('back_expiry')
        short_strike = signal.get('short_strike')
        long_strike = signal.get('long_strike')
        right = 'call' if 'CALL' in signal.get('signal','').upper() else 'put'

        # Fetch legs
        short_leg = options_df[(options_df['expiration']==front) & (options_df['strike']==short_strike) & (options_df['option_type']==right)]
        long_leg = options_df[(options_df['expiration']==back) & (options_df['strike']==long_strike) & (options_df['option_type']==right)]
        if short_leg.empty or long_leg.empty:
            return None

        short_price = float(short_leg.iloc[0].get('mid', short_leg.iloc[0].get('last', 0)))
        long_price = float(long_leg.iloc[0].get('mid', long_leg.iloc[0].get('last', 0)))
        if short_price<=0 or long_price<=0:
            return None

        net_debit = long_price - short_price
        T_front = (front - pd.Timestamp.now()).days/365
        T_back = (back - pd.Timestamp.now()).days/365
        iv_front = float(short_leg.iloc[0].get('impliedVolatility', historical_vol))
        iv_back = float(long_leg.iloc[0].get('impliedVolatility', historical_vol))

        # Greeks (approx): use ATM-ish
        g_short = self.greeks.all_greeks(current_price, short_strike, T_front, iv_front, right)
        g_long = self.greeks.all_greeks(current_price, long_strike, T_back, iv_back, right)
        greeks = {k: g_long[k]-g_short[k] for k in ['delta','gamma','theta','vega']}

        strategy = {
            'type': f"{right.upper()}_DIAGONAL",
            'signal_source': signal.get('type'),
            'short_expiration': front,
            'long_expiration': back,
            'short_strike': short_strike,
            'long_strike': long_strike,
            'days_to_expiry': int(T_front*365),
            'entry_cost': net_debit,
            'max_risk': net_debit,
            'max_profit': None,
            'implied_vol_front': iv_front,
            'implied_vol_back': iv_back,
            'greeks': greeks,
            'probability_profit': 0.0,
            'expected_value': 0.0,
            'expected_return': 0.0,
            'var_95': 0.0,
            'sharpe': 0.0,
            'strength_score': signal.get('seasonality_z', 0)*10 + 50,
            'confidence': 'MEDIUM',
            'recommendation': signal.get('recommendation','')
        }
        return strategy

    def _build_calendar_from_signal(
        self,
        signal: Dict,
        options_df: pd.DataFrame,
        current_price: float,
        historical_vol: float
    ) -> Optional[Dict]:
        """Build calendar spread with proper pricing"""

        strike = signal.get('strike')
        short_exp = signal.get('short_expiry')
        long_exp = signal.get('long_expiry')
        option_type = 'call'  # Default to call calendar

        # Get short option
        short_opt = options_df[
            (options_df['strike'] == strike) &
            (options_df['expiration'] == short_exp) &
            (options_df['option_type'] == option_type)
        ]

        # Get long option
        long_opt = options_df[
            (options_df['strike'] == strike) &
            (options_df['expiration'] == long_exp) &
            (options_df['option_type'] == option_type)
        ]

        if short_opt.empty or long_opt.empty:
            return None

        short_price = short_opt.iloc[0].get('mid', short_opt.iloc[0].get('last', 0))
        long_price = long_opt.iloc[0].get('mid', long_opt.iloc[0].get('last', 0))

        if short_price == 0 or long_price == 0:
            return None

        # Calendar spread: buy long, sell short (net debit)
        calendar_cost = long_price - short_price

        if calendar_cost <= 0:
            return None  # Should be debit spread

        # Get IVs
        short_iv = short_opt.iloc[0].get('impliedVolatility', historical_vol)
        long_iv = long_opt.iloc[0].get('impliedVolatility', historical_vol)

        # Time to expiry
        T_short = (short_exp - pd.Timestamp.now()).days / 365
        T_long = (long_exp - pd.Timestamp.now()).days / 365

        # Monte Carlo simulation
        mc_results = self._monte_carlo_calendar(
            current_price, strike, short_price, long_price,
            T_short, T_long, short_iv, long_iv, historical_vol
        )

        strategy = {
            'type': 'CALENDAR_SPREAD',
            'signal_source': signal.get('type'),
            'strike': strike,
            'short_expiry': short_exp,
            'long_expiry': long_exp,
            'days_to_short_expiry': int(T_short * 365),
            'days_to_long_expiry': int(T_long * 365),
            'entry_cost': calendar_cost,
            'short_price': short_price,
            'long_price': long_price,
            'short_iv': short_iv,
            'long_iv': long_iv,
            'max_risk': calendar_cost,
            'max_profit': mc_results['max_profit'],
            'probability_profit': mc_results['prob_profit'],
            'expected_value': mc_results['expected_value'],
            'expected_return': mc_results['expected_return_pct'],
            'var_95': mc_results['var_95'],
            'sharpe': mc_results['sharpe'],
            'strength_score': signal.get('strength_score', 50),
            'confidence': signal.get('confidence', 'MEDIUM'),
            'recommendation': signal.get('recommendation', '')
        }

        return strategy

    def _build_butterfly_from_signal(
        self,
        signal: Dict,
        options_df: pd.DataFrame,
        current_price: float,
        historical_vol: float
    ) -> Optional[Dict]:
        """Build butterfly spread with proper pricing"""

        expiry = signal.get('expiration')
        k_low = signal.get('strike_low')
        k_mid = signal.get('strike_mid')
        k_high = signal.get('strike_high')
        option_type = 'call'  # Default to call butterfly

        # Get options at each strike
        opt_low = options_df[
            (options_df['strike'] == k_low) &
            (options_df['expiration'] == expiry) &
            (options_df['option_type'] == option_type)
        ]

        opt_mid = options_df[
            (options_df['strike'] == k_mid) &
            (options_df['expiration'] == expiry) &
            (options_df['option_type'] == option_type)
        ]

        opt_high = options_df[
            (options_df['strike'] == k_high) &
            (options_df['expiration'] == expiry) &
            (options_df['option_type'] == option_type)
        ]

        if opt_low.empty or opt_mid.empty or opt_high.empty:
            return None

        price_low = opt_low.iloc[0].get('mid', opt_low.iloc[0].get('last', 0))
        price_mid = opt_mid.iloc[0].get('mid', opt_mid.iloc[0].get('last', 0))
        price_high = opt_high.iloc[0].get('mid', opt_high.iloc[0].get('last', 0))

        if price_low == 0 or price_mid == 0 or price_high == 0:
            return None

        # Butterfly cost: buy 1 low, sell 2 mid, buy 1 high (net debit)
        butterfly_cost = price_low - 2 * price_mid + price_high

        if butterfly_cost <= 0:
            return None  # Should be debit

        # Max profit at middle strike
        wing_width = k_mid - k_low
        max_profit = wing_width - butterfly_cost

        # Time to expiry
        T = (expiry - pd.Timestamp.now()).days / 365

        # Monte Carlo simulation
        mc_results = self._monte_carlo_butterfly(
            current_price, k_low, k_mid, k_high,
            butterfly_cost, T, historical_vol
        )

        strategy = {
            'type': 'BUTTERFLY',
            'signal_source': signal.get('type'),
            'expiration': expiry,
            'days_to_expiry': int(T * 365),
            'strike_low': k_low,
            'strike_mid': k_mid,
            'strike_high': k_high,
            'entry_cost': butterfly_cost,
            'max_risk': butterfly_cost,
            'max_profit': max_profit,
            'probability_profit': mc_results['prob_profit'],
            'expected_value': mc_results['expected_value'],
            'expected_return': mc_results['expected_return_pct'],
            'var_95': mc_results['var_95'],
            'sharpe': mc_results['sharpe'],
            'strength_score': signal.get('strength_score', 50),
            'confidence': signal.get('confidence', 'MEDIUM'),
            'recommendation': signal.get('recommendation', '')
        }

        return strategy

    def _build_from_term_signal(
        self,
        signal: Dict,
        options_df: pd.DataFrame,
        current_price: float,
        historical_vol: float
    ) -> Optional[Dict]:
        """Build strategy from term structure signal"""

        signal_type = signal.get('type')
        signal_action = signal.get('signal')

        # For VOL_SPIKE: suggest short front month straddle
        if signal_type == 'VOL_SPIKE':
            # Find shortest dated ATM straddle
            expiries = sorted(options_df['expiration'].unique())
            if len(expiries) == 0:
                return None

            front_month = expiries[0]

            # Find ATM strike
            front_opts = options_df[options_df['expiration'] == front_month]
            strikes = sorted(front_opts['strike'].unique())
            atm_strike = min(strikes, key=lambda x: abs(x - current_price))

            # Get straddle
            call = front_opts[
                (front_opts['strike'] == atm_strike) &
                (front_opts['option_type'] == 'call')
            ]
            put = front_opts[
                (front_opts['strike'] == atm_strike) &
                (front_opts['option_type'] == 'put')
            ]

            if not call.empty and not put.empty:
                call_price = call.iloc[0].get('mid', call.iloc[0].get('last', 0))
                put_price = put.iloc[0].get('mid', put.iloc[0].get('last', 0))

                if call_price > 0 and put_price > 0:
                    straddle_price = call_price + put_price
                    T = (front_month - pd.Timestamp.now()).days / 365

                    # Use short straddle Monte Carlo
                    front_iv = call.iloc[0].get('impliedVolatility', historical_vol)
                    mc_results = self._monte_carlo_straddle(
                        current_price, atm_strike, straddle_price, T, historical_vol, front_iv, is_long=False
                    )

                    strategy = {
                        'type': 'SHORT_STRADDLE',
                        'signal_source': signal_type,
                        'strike': atm_strike,
                        'expiration': front_month,
                        'days_to_expiry': int(T * 365),
                        'entry_cost': -straddle_price,
                        'max_risk': np.inf,
                        'max_profit': straddle_price,
                        'probability_profit': mc_results['prob_profit'],
                        'expected_value': mc_results['expected_value'],
                        'expected_return': mc_results['expected_return_pct'],
                        'var_95': mc_results['var_95'],
                        'sharpe': mc_results['sharpe'],
                        'strength_score': signal.get('strength_score', 50),
                        'confidence': signal.get('confidence', 'MEDIUM'),
                        'recommendation': signal.get('recommendation', '')
                    }

                    return strategy

        # For generic TERM_STRUCTURE signals: simplified placeholder
        strategy = {
            'type': signal.get('signal', 'TERM_PLAY'),
            'signal_source': signal.get('type'),
            'action': signal.get('action', ''),
            'strength_score': signal.get('strength_score', 50),
            'confidence': signal.get('confidence', 'LOW'),
            'recommendation': signal.get('recommendation', ''),
            'expected_value': 0,
            'max_risk': 100,
            'probability_profit': 0.5
        }

        return strategy

    def _monte_carlo_straddle(
        self,
        S: float,
        K: float,
        straddle_price: float,
        T: float,
        realized_vol: float,
        implied_vol: float,
        is_long: bool,
        n_sims: int = 10000
    ) -> Dict:
        """
        Monte Carlo simulation for straddle

        Uses realized_vol for price simulations (HV), implied_vol embedded in prices.
        Edge comes from IV-HV mismatch.

        Returns:
            Dict with probability, expected value, VaR, Sharpe, etc.
        """
        # Simulate price paths using realized volatility
        np.random.seed(42)  # For reproducibility

        # GBM simulation with realized vol (what actually happens)
        drift = -0.5 * realized_vol ** 2  # Risk-neutral drift
        shock = realized_vol * np.sqrt(T) * np.random.randn(n_sims)

        final_prices = S * np.exp(drift * T + shock)

        # Straddle payoff at expiry
        payoffs = np.maximum(final_prices - K, 0) + np.maximum(K - final_prices, 0)

        # P&L (straddle_price includes implied vol pricing)
        if is_long:
            pnl = payoffs - straddle_price
        else:
            pnl = straddle_price - payoffs

        # Metrics
        prob_profit = (pnl > 0).sum() / n_sims
        expected_value = pnl.mean()
        std_dev = pnl.std()
        var_95 = np.percentile(pnl, 5)
        sharpe = expected_value / std_dev if std_dev > 0 else 0

        expected_return_pct = (expected_value / straddle_price) * 100 if straddle_price > 0 else 0

        return {
            'prob_profit': prob_profit,
            'expected_value': expected_value,
            'expected_return_pct': expected_return_pct,
            'std_dev': std_dev,
            'var_95': var_95,
            'sharpe': sharpe
        }

    def _monte_carlo_strangle(
        self,
        S: float,
        K_put: float,
        K_call: float,
        strangle_price: float,
        T: float,
        realized_vol: float,
        implied_vol: float,
        is_long: bool,
        n_sims: int = 10000
    ) -> Dict:
        """Monte Carlo for strangle - uses realized vol for simulations"""

        np.random.seed(42)

        # Use realized volatility for price simulations
        drift = -0.5 * realized_vol ** 2
        shock = realized_vol * np.sqrt(T) * np.random.randn(n_sims)

        final_prices = S * np.exp(drift * T + shock)

        # Strangle payoff
        payoffs = np.maximum(final_prices - K_call, 0) + np.maximum(K_put - final_prices, 0)

        if is_long:
            pnl = payoffs - strangle_price
        else:
            pnl = strangle_price - payoffs

        prob_profit = (pnl > 0).sum() / n_sims
        expected_value = pnl.mean()
        std_dev = pnl.std()
        var_95 = np.percentile(pnl, 5)
        sharpe = expected_value / std_dev if std_dev > 0 else 0

        expected_return_pct = (expected_value / strangle_price) * 100 if strangle_price > 0 else 0

        return {
            'prob_profit': prob_profit,
            'expected_value': expected_value,
            'expected_return_pct': expected_return_pct,
            'std_dev': std_dev,
            'var_95': var_95,
            'sharpe': sharpe
        }

    def _monte_carlo_calendar(
        self,
        S: float,
        K: float,
        short_price: float,
        long_price: float,
        T_short: float,
        T_long: float,
        short_iv: float,
        long_iv: float,
        vol: float,
        n_sims: int = 10000
    ) -> Dict:
        """
        Monte Carlo simulation for calendar spread

        Approximates by simulating price at short expiry, then pricing long option
        """
        np.random.seed(42)

        calendar_cost = long_price - short_price

        # Simulate price at short expiry
        drift = -0.5 * vol ** 2
        shock = vol * np.sqrt(T_short) * np.random.randn(n_sims)
        prices_at_short_exp = S * np.exp(drift * T_short + shock)

        # Short option payoff at expiry (we sold it)
        short_payoffs = np.maximum(prices_at_short_exp - K, 0)

        # Long option value at short expiry (time value remaining)
        # Remaining time for long option
        T_remaining = T_long - T_short

        # Approximate long option value using Black-Scholes
        long_values = np.zeros(n_sims)
        for i, S_t in enumerate(prices_at_short_exp):
            if T_remaining > 0:
                # Simplified: use current greeks calculator
                try:
                    long_val = self.greeks.black_scholes_price(S_t, K, T_remaining, long_iv, 'call')
                    long_values[i] = long_val
                except:
                    long_values[i] = max(S_t - K, 0)  # Intrinsic value fallback
            else:
                long_values[i] = max(S_t - K, 0)

        # P&L: (long option value at short expiry) + (short option premium) - (initial cost)
        pnl = long_values - short_payoffs - calendar_cost

        # Metrics
        prob_profit = (pnl > 0).sum() / n_sims
        expected_value = pnl.mean()
        std_dev = pnl.std()
        var_95 = np.percentile(pnl, 5)
        sharpe = expected_value / std_dev if std_dev > 0 else 0
        expected_return_pct = (expected_value / calendar_cost) * 100 if calendar_cost > 0 else 0

        # Max profit approximately when price stays at strike
        max_profit = long_price

        return {
            'prob_profit': prob_profit,
            'expected_value': expected_value,
            'expected_return_pct': expected_return_pct,
            'std_dev': std_dev,
            'var_95': var_95,
            'sharpe': sharpe,
            'max_profit': max_profit
        }

    def _monte_carlo_butterfly(
        self,
        S: float,
        K_low: float,
        K_mid: float,
        K_high: float,
        butterfly_cost: float,
        T: float,
        vol: float,
        n_sims: int = 10000
    ) -> Dict:
        """
        Monte Carlo simulation for butterfly spread

        Butterfly: Buy 1 low, Sell 2 mid, Buy 1 high
        """
        np.random.seed(42)

        # Simulate price at expiry
        drift = -0.5 * vol ** 2
        shock = vol * np.sqrt(T) * np.random.randn(n_sims)
        final_prices = S * np.exp(drift * T + shock)

        # Butterfly payoff at expiry
        payoffs = np.zeros(n_sims)
        for i, S_T in enumerate(final_prices):
            # Long 1 call at K_low
            payoff_low = max(S_T - K_low, 0)
            # Short 2 calls at K_mid
            payoff_mid = -2 * max(S_T - K_mid, 0)
            # Long 1 call at K_high
            payoff_high = max(S_T - K_high, 0)

            payoffs[i] = payoff_low + payoff_mid + payoff_high

        # P&L
        pnl = payoffs - butterfly_cost

        # Metrics
        prob_profit = (pnl > 0).sum() / n_sims
        expected_value = pnl.mean()
        std_dev = pnl.std()
        var_95 = np.percentile(pnl, 5)
        sharpe = expected_value / std_dev if std_dev > 0 else 0
        expected_return_pct = (expected_value / butterfly_cost) * 100 if butterfly_cost > 0 else 0

        return {
            'prob_profit': prob_profit,
            'expected_value': expected_value,
            'expected_return_pct': expected_return_pct,
            'std_dev': std_dev,
            'var_95': var_95,
            'sharpe': sharpe
        }


if __name__ == "__main__":
    # Test optimizer
    print("Testing Strategy Optimizer\n")

    from data_manager import OptionsDataManager
    from alpha_screener import AlphaScreener
    from vol_arbitrage import VolatilityArbitrageStrategies

    data_mgr = OptionsDataManager()
    greeks = GreeksCalculator()
    vol_arb = VolatilityArbitrageStrategies(greeks)
    screener = AlphaScreener(greeks, vol_arb)
    optimizer = StrategyOptimizer(greeks)

    # Test with AAPL
    ticker = 'AAPL'
    print(f"Analyzing {ticker}...\n")

    options_df = data_mgr.fetch_options_data(ticker, 'yfinance')
    current_price = data_mgr.get_current_price(ticker)
    hv = data_mgr.calculate_historical_volatility(ticker)

    # Get signals
    signals = screener.screen_opportunities(options_df, current_price, hv, ticker)

    print(f"Screening found {len(signals)} signals")

    # Rank strategies
    strategies = optimizer.rank_strategies(options_df, current_price, hv, signals)

    print(f"\nTop 5 strategies:\n")

    for i, strat in enumerate(strategies[:5], 1):
        print(f"{i}. {strat['type']}")
        print(f"   Expected Return: {strat.get('expected_return', 0):.2f}%")
        print(f"   Prob Profit: {strat.get('probability_profit', 0)*100:.1f}%")
        print(f"   Max Risk: ${strat.get('max_risk', 0):.2f}")
        print(f"   Sharpe: {strat.get('sharpe', 0):.2f}")
        print()

    print("Strategy Optimizer test complete!")
