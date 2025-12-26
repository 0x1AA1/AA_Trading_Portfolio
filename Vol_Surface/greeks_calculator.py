"""
Options Greeks Calculator and Advanced Pricing Models

Includes:
- Black-Scholes Greeks (Delta, Gamma, Vega, Theta, Rho)
- Straddle/Strangle pricing and analysis
- Break-even calculations
- Profit/Loss distributions
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd


class GreeksCalculator:
    """
    Comprehensive options Greeks calculator

    Based on Black-Scholes framework with extensions
    """

    def __init__(self, risk_free_rate=0.05, dividend_yield=0.0):
        """
        Args:
            risk_free_rate: Annual risk-free rate
            dividend_yield: Annual continuous dividend yield
        """
        self.r = risk_free_rate
        self.q = dividend_yield

    def _d1_d2(self, S, K, T, sigma):
        """
        Calculate d1 and d2 for Black-Scholes formula

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            sigma: Volatility

        Returns:
            d1, d2: tuple
        """
        if T <= 0:
            return 0, 0

        d1 = (np.log(S / K) + (self.r - self.q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        return d1, d2

    def black_scholes_price(self, S, K, T, sigma, option_type='call'):
        """
        Black-Scholes option price

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            sigma: Volatility
            option_type: 'call' or 'put'

        Returns:
            price: Option price
        """
        if T <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        d1, d2 = self._d1_d2(S, K, T, sigma)

        if option_type == 'call':
            price = S * np.exp(-self.q * T) * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-self.r * T) * norm.cdf(-d2) - S * np.exp(-self.q * T) * norm.cdf(-d1)

        return price

    def delta(self, S, K, T, sigma, option_type='call'):
        """
        Delta: Rate of change of option price with respect to underlying price

        ∂V/∂S
        """
        if T <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0

        d1, _ = self._d1_d2(S, K, T, sigma)

        if option_type == 'call':
            return np.exp(-self.q * T) * norm.cdf(d1)
        else:  # put
            return -np.exp(-self.q * T) * norm.cdf(-d1)

    def gamma(self, S, K, T, sigma):
        """
        Gamma: Rate of change of delta with respect to underlying price

        ∂²V/∂S²

        Same for calls and puts
        """
        if T <= 0:
            return 0.0

        d1, _ = self._d1_d2(S, K, T, sigma)

        gamma = (np.exp(-self.q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))

        return gamma

    def vega(self, S, K, T, sigma):
        """
        Vega: Rate of change of option price with respect to volatility

        ∂V/∂σ

        Same for calls and puts
        Typically expressed as change per 1% change in volatility
        """
        if T <= 0:
            return 0.0

        d1, _ = self._d1_d2(S, K, T, sigma)

        vega = S * np.exp(-self.q * T) * norm.pdf(d1) * np.sqrt(T)

        # Convert to per 1% volatility change
        return vega / 100

    def theta(self, S, K, T, sigma, option_type='call'):
        """
        Theta: Rate of change of option price with respect to time

        ∂V/∂t (negative of this, actually)

        Typically expressed as decay per day
        """
        if T <= 0:
            return 0.0

        d1, d2 = self._d1_d2(S, K, T, sigma)

        term1 = -(S * norm.pdf(d1) * sigma * np.exp(-self.q * T)) / (2 * np.sqrt(T))

        if option_type == 'call':
            term2 = self.q * S * norm.cdf(d1) * np.exp(-self.q * T)
            term3 = self.r * K * np.exp(-self.r * T) * norm.cdf(d2)
            theta = term1 - term2 + term3
        else:  # put
            term2 = self.q * S * norm.cdf(-d1) * np.exp(-self.q * T)
            term3 = self.r * K * np.exp(-self.r * T) * norm.cdf(-d2)
            theta = term1 + term2 - term3

        # Convert to per-day decay
        return theta / 365

    def rho(self, S, K, T, sigma, option_type='call'):
        """
        Rho: Rate of change of option price with respect to risk-free rate

        ∂V/∂r

        Typically expressed as change per 1% change in interest rate
        """
        if T <= 0:
            return 0.0

        _, d2 = self._d1_d2(S, K, T, sigma)

        if option_type == 'call':
            rho = K * T * np.exp(-self.r * T) * norm.cdf(d2)
        else:  # put
            rho = -K * T * np.exp(-self.r * T) * norm.cdf(-d2)

        # Convert to per 1% rate change
        return rho / 100

    def all_greeks(self, S, K, T, sigma, option_type='call'):
        """
        Calculate all Greeks at once

        Returns:
            dict with all Greeks
        """
        return {
            'price': self.black_scholes_price(S, K, T, sigma, option_type),
            'delta': self.delta(S, K, T, sigma, option_type),
            'gamma': self.gamma(S, K, T, sigma),
            'vega': self.vega(S, K, T, sigma),
            'theta': self.theta(S, K, T, sigma, option_type),
            'rho': self.rho(S, K, T, sigma, option_type)
        }


class StraddleStrangle:
    """
    Straddle and Strangle pricing and analysis
    """

    def __init__(self, greeks_calculator):
        """
        Args:
            greeks_calculator: GreeksCalculator instance
        """
        self.calc = greeks_calculator

    def straddle_price(self, S, K, T, sigma):
        """
        Price of a straddle (long call + long put at same strike)

        Args:
            S: Spot price
            K: Strike price (typically ATM)
            T: Time to maturity
            sigma: Implied volatility

        Returns:
            total_price: Straddle price
            components: dict with call and put prices
        """
        call_price = self.calc.black_scholes_price(S, K, T, sigma, 'call')
        put_price = self.calc.black_scholes_price(S, K, T, sigma, 'put')

        return call_price + put_price, {'call': call_price, 'put': put_price}

    def strangle_price(self, S, K_call, K_put, T, sigma):
        """
        Price of a strangle (long call at K_call + long put at K_put)

        Typically K_put < S < K_call

        Args:
            S: Spot price
            K_call: Call strike (above S)
            K_put: Put strike (below S)
            T: Time to maturity
            sigma: Implied volatility

        Returns:
            total_price: Strangle price
            components: dict with call and put prices
        """
        call_price = self.calc.black_scholes_price(S, K_call, T, sigma, 'call')
        put_price = self.calc.black_scholes_price(S, K_put, T, sigma, 'put')

        return call_price + put_price, {'call': call_price, 'put': put_price}

    def straddle_breakeven(self, S, K, T, sigma):
        """
        Calculate breakeven points for a long straddle

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            sigma: Implied volatility

        Returns:
            lower_breakeven, upper_breakeven: tuple
        """
        straddle_cost, _ = self.straddle_price(S, K, T, sigma)

        lower_breakeven = K - straddle_cost
        upper_breakeven = K + straddle_cost

        return lower_breakeven, upper_breakeven

    def straddle_greeks(self, S, K, T, sigma):
        """
        Calculate Greeks for a straddle position

        Returns:
            dict with combined Greeks
        """
        call_greeks = self.calc.all_greeks(S, K, T, sigma, 'call')
        put_greeks = self.calc.all_greeks(S, K, T, sigma, 'put')

        return {
            'price': call_greeks['price'] + put_greeks['price'],
            'delta': call_greeks['delta'] + put_greeks['delta'],  # Should be ~0 for ATM
            'gamma': call_greeks['gamma'] + put_greeks['gamma'],
            'vega': call_greeks['vega'] + put_greeks['vega'],
            'theta': call_greeks['theta'] + put_greeks['theta'],
            'rho': call_greeks['rho'] + put_greeks['rho']
        }

    def pnl_distribution(self, S, K, T, sigma, spot_range=None, n_points=100):
        """
        Calculate P&L distribution for long straddle at expiration

        Args:
            S: Current spot price
            K: Strike price
            T: Time to maturity
            sigma: Implied volatility
            spot_range: (min_spot, max_spot) or None for auto
            n_points: Number of points to calculate

        Returns:
            DataFrame with spot prices and P&L
        """
        straddle_cost, _ = self.straddle_price(S, K, T, sigma)

        if spot_range is None:
            # Auto-range: ±3 standard deviations
            std = sigma * S * np.sqrt(T)
            spot_range = (max(S - 3*std, 0.01), S + 3*std)

        spot_prices = np.linspace(spot_range[0], spot_range[1], n_points)

        # P&L at expiration
        pnl = np.maximum(spot_prices - K, 0) + np.maximum(K - spot_prices, 0) - straddle_cost

        # Current P&L (mark-to-market)
        current_pnl = []
        for spot in spot_prices:
            current_value, _ = self.straddle_price(spot, K, T, sigma)
            current_pnl.append(current_value - straddle_cost)

        return pd.DataFrame({
            'spot_price': spot_prices,
            'pnl_expiration': pnl,
            'pnl_current': current_pnl,
            'probability': norm.pdf(np.log(spot_prices/S),
                                   (self.calc.r - 0.5*sigma**2)*T,
                                   sigma*np.sqrt(T)) / spot_prices
        })

    def optimal_strike_analysis(self, S, T, sigma, strike_range=None, n_strikes=50):
        """
        Analyze straddle pricing across different strikes

        Args:
            S: Spot price
            T: Time to maturity
            sigma: Implied volatility
            strike_range: (min_strike, max_strike) or None for auto
            n_strikes: Number of strikes to analyze

        Returns:
            DataFrame with strike analysis
        """
        if strike_range is None:
            strike_range = (S * 0.7, S * 1.3)

        strikes = np.linspace(strike_range[0], strike_range[1], n_strikes)

        results = []
        for K in strikes:
            price, components = self.straddle_price(S, K, T, sigma)
            greeks = self.straddle_greeks(S, K, T, sigma)
            lower_be, upper_be = self.straddle_breakeven(S, K, T, sigma)

            results.append({
                'strike': K,
                'moneyness': K / S,
                'straddle_price': price,
                'call_price': components['call'],
                'put_price': components['put'],
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'vega': greeks['vega'],
                'theta': greeks['theta'],
                'lower_breakeven': lower_be,
                'upper_breakeven': upper_be,
                'breakeven_range': upper_be - lower_be,
                'required_move_pct': abs(K - S) / S * 100
            })

        return pd.DataFrame(results)


class VolatilityMetrics:
    """
    Advanced volatility surface metrics and analysis
    """

    @staticmethod
    def skew_25delta(iv_25put, iv_atm, iv_25call):
        """
        25-delta risk reversal (skew indicator)

        Args:
            iv_25put: IV of 25-delta put
            iv_atm: IV of ATM option
            iv_25call: IV of 25-delta call

        Returns:
            risk_reversal: 25-delta RR (put IV - call IV)
            butterfly: 25-delta butterfly spread
        """
        risk_reversal = iv_25put - iv_25call
        butterfly = (iv_25put + iv_25call) / 2 - iv_atm

        return risk_reversal, butterfly

    @staticmethod
    def implied_vol_smile(strikes, spot, ivs):
        """
        Characterize volatility smile

        Args:
            strikes: Array of strike prices
            spot: Spot price
            ivs: Array of implied volatilities

        Returns:
            dict with smile characteristics
        """
        moneyness = strikes / spot

        # Find ATM IV
        atm_idx = np.argmin(np.abs(moneyness - 1.0))
        atm_iv = ivs[atm_idx]

        # Slope on both sides
        left_mask = moneyness < 1.0
        right_mask = moneyness > 1.0

        if np.sum(left_mask) > 1:
            left_slope = np.polyfit(moneyness[left_mask], ivs[left_mask], 1)[0]
        else:
            left_slope = 0.0

        if np.sum(right_mask) > 1:
            right_slope = np.polyfit(moneyness[right_mask], ivs[right_mask], 1)[0]
        else:
            right_slope = 0.0

        # Convexity (second derivative)
        if len(ivs) >= 3:
            second_deriv = np.gradient(np.gradient(ivs))
            avg_convexity = np.mean(second_deriv)
        else:
            avg_convexity = 0.0

        return {
            'atm_iv': atm_iv,
            'left_slope': left_slope,
            'right_slope': right_slope,
            'skew': left_slope - right_slope,
            'convexity': avg_convexity,
            'smile_shape': 'symmetric' if abs(left_slope - right_slope) < 0.1 else 'skewed'
        }

    @staticmethod
    def term_structure_slope(maturities, atm_ivs):
        """
        Analyze term structure of ATM volatility

        Args:
            maturities: Array of maturities (years)
            atm_ivs: Array of ATM implied volatilities

        Returns:
            dict with term structure characteristics
        """
        if len(maturities) < 2:
            return {'slope': 0.0, 'shape': 'flat'}

        # Linear fit
        slope, intercept = np.polyfit(maturities, atm_ivs, 1)

        # Determine shape
        if slope > 0.05:
            shape = 'upward_sloping'
        elif slope < -0.05:
            shape = 'downward_sloping'
        else:
            shape = 'flat'

        return {
            'slope': slope,
            'intercept': intercept,
            'shape': shape,
            'short_term_iv': atm_ivs[0] if len(atm_ivs) > 0 else None,
            'long_term_iv': atm_ivs[-1] if len(atm_ivs) > 0 else None
        }
