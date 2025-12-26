"""
Volatility Arbitrage Strategies

Based on research from:
- Carr & Madan (1998) - "Towards a Theory of Volatility Trading"
- Carr et al. (2005) - "Pricing Options on Realized Variance"
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from greeks_calculator import GreeksCalculator


class VolatilityArbitrageStrategies:
    """
    Implement various volatility arbitrage trading strategies
    """

    def __init__(self, greeks_calc):
        """
        Args:
            greeks_calc: GreeksCalculator instance
        """
        self.calc = greeks_calc

    def calendar_spread_arbitrage(self, S, K, T_short, T_long, sigma_short, sigma_long):
        """
        Calendar spread arbitrage detector

        Total variance should be increasing in maturity:
        sigma_short^2 * T_short <= sigma_long^2 * T_long

        Args:
            S: Spot price
            K: Strike price
            T_short: Short maturity
            T_long: Long maturity
            sigma_short: IV for short maturity
            sigma_long: IV for long maturity

        Returns:
            dict with arbitrage detection results
        """
        total_var_short = sigma_short**2 * T_short
        total_var_long = sigma_long**2 * T_long

        arbitrage = total_var_long < total_var_short

        # Calendar spread price (long long-term, short short-term)
        call_short = self.calc.black_scholes_price(S, K, T_short, sigma_short, 'call')
        call_long = self.calc.black_scholes_price(S, K, T_long, sigma_long, 'call')

        spread_price = call_long - call_short

        return {
            'arbitrage_detected': arbitrage,
            'total_var_short': total_var_short,
            'total_var_long': total_var_long,
            'spread_price': spread_price,
            'forward_variance': (total_var_long - total_var_short) / (T_long - T_short) if T_long > T_short else 0,
            'recommendation': 'BUY CALENDAR SPREAD' if spread_price < 0 and not arbitrage else 'NO ARBITRAGE'
        }

    def butterfly_arbitrage(self, S, K_low, K_mid, K_high, T, sigma_low, sigma_mid, sigma_high):
        """
        Butterfly spread arbitrage detector

        Implied variance should be convex in strike:
        sigma_mid^2 <= (sigma_low^2 + sigma_high^2) / 2

        Args:
            S: Spot price
            K_low, K_mid, K_high: Strike prices (equally spaced recommended)
            T: Time to maturity
            sigma_low, sigma_mid, sigma_high: IVs at respective strikes

        Returns:
            dict with arbitrage detection results
        """
        # Check convexity in total variance
        var_low = sigma_low**2
        var_mid = sigma_mid**2
        var_high = sigma_high**2

        convexity_satisfied = var_mid <= (var_low + var_high) / 2
        arbitrage = not convexity_satisfied

        # Butterfly price: long 1 low strike, long 1 high strike, short 2 mid strike
        call_low = self.calc.black_scholes_price(S, K_low, T, sigma_low, 'call')
        call_mid = self.calc.black_scholes_price(S, K_mid, T, sigma_mid, 'call')
        call_high = self.calc.black_scholes_price(S, K_high, T, sigma_high, 'call')

        butterfly_price = call_low + call_high - 2 * call_mid

        return {
            'arbitrage_detected': arbitrage,
            'convexity_satisfied': convexity_satisfied,
            'var_low': var_low,
            'var_mid': var_mid,
            'var_high': var_high,
            'butterfly_price': butterfly_price,
            'max_payout': (K_high - K_low) if K_mid == (K_low + K_high) / 2 else 0,
            'recommendation': 'BUY BUTTERFLY' if butterfly_price < 0 and arbitrage else 'NO ARBITRAGE'
        }

    def volatility_spread_strategy(self, S, K, T, implied_vol, historical_vol, threshold=0.10):
        """
        Compare implied vs historical volatility for trading signal

        Strategy:
        - If IV > HV + threshold: Sell volatility (short straddle)
        - If IV < HV - threshold: Buy volatility (long straddle)

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            implied_vol: Market implied volatility
            historical_vol: Historical realized volatility
            threshold: Minimum spread to trigger signal

        Returns:
            dict with strategy recommendation
        """
        vol_spread = implied_vol - historical_vol
        vol_spread_pct = vol_spread / historical_vol * 100

        # Straddle price at implied vol
        call_price = self.calc.black_scholes_price(S, K, T, implied_vol, 'call')
        put_price = self.calc.black_scholes_price(S, K, T, implied_vol, 'put')
        straddle_price = call_price + put_price

        # Expected P&L if realized vol equals historical vol
        # Simplified: gamma profit vs theta decay
        gamma = self.calc.all_greeks(S, K, T, implied_vol, 'call')['gamma']
        theta = (self.calc.all_greeks(S, K, T, implied_vol, 'call')['theta'] +
                self.calc.all_greeks(S, K, T, implied_vol, 'put')['theta'])

        # Expected variance profit: 0.5 * gamma * S^2 * (realized_var - implied_var)
        realized_var = historical_vol**2
        implied_var = implied_vol**2
        var_profit_per_day = 0.5 * gamma * S**2 * (realized_var - implied_var) / 252

        if vol_spread > threshold:
            signal = 'SELL_VOLATILITY'
            action = 'Short straddle'
            expected_pnl = -var_profit_per_day * T * 252 + theta * T * 252
        elif vol_spread < -threshold:
            signal = 'BUY_VOLATILITY'
            action = 'Long straddle'
            expected_pnl = var_profit_per_day * T * 252 - theta * T * 252
        else:
            signal = 'NO_TRADE'
            action = 'No clear edge'
            expected_pnl = 0

        return {
            'signal': signal,
            'action': action,
            'implied_vol': implied_vol,
            'historical_vol': historical_vol,
            'vol_spread': vol_spread,
            'vol_spread_pct': vol_spread_pct,
            'straddle_price': straddle_price,
            'expected_pnl': expected_pnl,
            'gamma': gamma,
            'theta_per_day': theta,
            'confidence': 'HIGH' if abs(vol_spread) > 2 * threshold else 'MEDIUM' if abs(vol_spread) > threshold else 'LOW'
        }

    def dispersion_trading_signal(self, index_iv, component_ivs, component_weights=None):
        """
        Dispersion trading: exploit difference between index volatility and
        weighted average of component volatilities

        Strategy:
        - If index IV < weighted component IV: Buy index vol, sell component vols
        - If index IV > weighted component IV: Sell index vol, buy component vols

        Args:
            index_iv: Index implied volatility
            component_ivs: Array of component IVs
            component_weights: Array of component weights (or None for equal weight)

        Returns:
            dict with dispersion trading signal
        """
        if component_weights is None:
            component_weights = np.ones(len(component_ivs)) / len(component_ivs)

        # Weighted average of component vols
        weighted_comp_iv = np.sum(np.array(component_ivs) * np.array(component_weights))

        # Dispersion
        dispersion = weighted_comp_iv - index_iv
        dispersion_pct = dispersion / index_iv * 100

        # Correlation implied by dispersion
        # Simplified relationship: index_var ≈ Σw_i^2 * var_i + ΣΣw_i*w_j*corr_ij*σ_i*σ_j
        # For rough estimate, assume equal correlation ρ

        if dispersion > 0.02:  # Components more expensive than index
            signal = 'LONG_DISPERSION'
            action = 'Buy index vol, sell component vols'
        elif dispersion < -0.02:
            signal = 'SHORT_DISPERSION'
            action = 'Sell index vol, buy component vols'
        else:
            signal = 'NO_TRADE'
            action = 'No clear dispersion edge'

        return {
            'signal': signal,
            'action': action,
            'index_iv': index_iv,
            'weighted_component_iv': weighted_comp_iv,
            'dispersion': dispersion,
            'dispersion_pct': dispersion_pct,
            'n_components': len(component_ivs),
            'component_iv_range': (min(component_ivs), max(component_ivs)),
            'recommendation': action
        }

    def vol_term_structure_trade(self, S, K, maturities, ivs, spot_vol_estimate):
        """
        Identify trading opportunities in volatility term structure

        Args:
            S: Spot price
            K: Strike price
            maturities: Array of maturities (years)
            ivs: Array of implied volatilities
            spot_vol_estimate: Estimated spot/short-term volatility

        Returns:
            dict with term structure analysis and trade recommendations
        """
        if len(maturities) < 2:
            return {'error': 'Need at least 2 maturities'}

        # Calculate term structure slope
        slope, intercept = np.polyfit(maturities, ivs, 1)

        # Compare short-term IV to spot vol estimate
        short_term_iv = ivs[0]
        long_term_iv = ivs[-1]

        trades = []

        # Steep contango
        if slope > 0.1 and short_term_iv < spot_vol_estimate:
            trades.append({
                'type': 'SELL_CALENDAR_SPREAD',
                'rationale': 'Steep contango, short-term undervalued',
                'action': 'Short long-term, long short-term'
            })

        # Steep backwardation
        if slope < -0.1 and long_term_iv < spot_vol_estimate:
            trades.append({
                'type': 'BUY_CALENDAR_SPREAD',
                'rationale': 'Steep backwardation, long-term undervalued',
                'action': 'Long long-term, short short-term'
            })

        # Mean reversion play
        if short_term_iv > long_term_iv * 1.3:
            trades.append({
                'type': 'FADE_SHORT_TERM',
                'rationale': 'Short-term vol spike likely to mean-revert',
                'action': 'Sell short-term volatility'
            })

        return {
            'slope': slope,
            'intercept': intercept,
            'short_term_iv': short_term_iv,
            'long_term_iv': long_term_iv,
            'spot_vol_estimate': spot_vol_estimate,
            'term_structure_shape': 'CONTANGO' if slope > 0.05 else 'BACKWARDATION' if slope < -0.05 else 'FLAT',
            'trades': trades,
            'n_opportunities': len(trades)
        }

    def variance_swap_replication(self, S, K_strikes, T, ivs, n_options=50):
        """
        Price a variance swap using a portfolio of vanilla options

        Based on Carr & Madan formula:
        Var_swap = (2/T) * ∫[C(K)/K^2]dK for K > S + ∫[P(K)/K^2]dK for K < S

        Args:
            S: Spot price
            K_strikes: Array of strike prices
            T: Maturity
            ivs: Array of implied volatilities at each strike
            n_options: Number of options to use in replication

        Returns:
            dict with variance swap valuation
        """
        # Separate calls and puts
        call_strikes = K_strikes[K_strikes > S]
        put_strikes = K_strikes[K_strikes <= S]

        call_ivs = ivs[K_strikes > S]
        put_ivs = ivs[K_strikes <= S]

        # Price options
        call_integral = 0
        for K, iv in zip(call_strikes, call_ivs):
            call_price = self.calc.black_scholes_price(S, K, T, iv, 'call')
            dK = call_strikes[1] - call_strikes[0] if len(call_strikes) > 1 else 1
            call_integral += call_price / K**2 * dK

        put_integral = 0
        for K, iv in zip(put_strikes, put_ivs):
            put_price = self.calc.black_scholes_price(S, K, T, iv, 'put')
            dK = put_strikes[1] - put_strikes[0] if len(put_strikes) > 1 else 1
            put_integral += put_price / K**2 * dK

        # Variance swap strike
        var_strike = (2 / T) * (call_integral + put_integral)

        # Annualized volatility strike
        vol_strike = np.sqrt(var_strike)

        return {
            'variance_strike': var_strike,
            'volatility_strike': vol_strike,
            'annualized_vol_strike': vol_strike * np.sqrt(252 / T) if T > 0 else 0,
            'n_calls': len(call_strikes),
            'n_puts': len(put_strikes),
            'call_integral': call_integral,
            'put_integral': put_integral,
            'replication_error': 'Low' if len(K_strikes) > 20 else 'Medium' if len(K_strikes) > 10 else 'High'
        }
