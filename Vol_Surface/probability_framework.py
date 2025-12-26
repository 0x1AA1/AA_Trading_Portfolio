"""
Probability Framework for Options Analysis

Statistical probability calculations including:
- Standard deviation-based price ranges
- Probability of reaching specific strikes
- Monte Carlo expected value analysis
- Time-weighted probability adjustments

Author: Aoures ABDI
Version: 1.0
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ProbabilityAnalyzer:
    """
    Statistical probability analysis for options trading decisions
    """

    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize probability analyzer

        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate

    def analyze_comprehensive_probabilities(
        self,
        current_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        option_premium: Optional[float] = None,
        option_type: str = 'call'
    ) -> Dict:
        """
        Complete probability analysis in one call

        Args:
            current_price: Current stock price
            strike: Option strike price
            time_to_expiry: Time to expiration (years)
            volatility: Annualized volatility
            option_premium: Option price (for expected value calc)
            option_type: 'call' or 'put'

        Returns:
            Comprehensive probability dictionary
        """
        logger.info(f"Analyzing probabilities for ${strike} {option_type}, "
                   f"T={time_to_expiry:.2f}y, σ={volatility*100:.1f}%")

        analysis = {
            'strike': strike,
            'current_price': current_price,
            'time_to_expiry_years': time_to_expiry,
            'time_to_expiry_days': int(time_to_expiry * 365),
            'volatility': volatility,

            # Standard deviation ranges
            'std_ranges': self.calculate_std_ranges(
                current_price, volatility, time_to_expiry
            ),

            # Strike probabilities
            'strike_probabilities': self.calculate_strike_probabilities(
                current_price, strike, time_to_expiry, volatility
            ),

            # Breakeven analysis
            'breakeven_analysis': self.calculate_breakeven_probability(
                current_price, strike, option_premium, time_to_expiry,
                volatility, option_type
            ) if option_premium else None
        }

        # Monte Carlo if premium provided
        if option_premium:
            analysis['expected_value_analysis'] = self.monte_carlo_expected_value(
                current_price, strike, time_to_expiry, volatility,
                option_premium, option_type
            )

        return analysis

    def calculate_std_ranges(
        self,
        current_price: float,
        volatility: float,
        time_to_expiry: float,
        n_std: int = 3
    ) -> Dict:
        """
        Calculate price ranges based on standard deviations

        Args:
            current_price: Current stock price
            volatility: Annualized volatility
            time_to_expiry: Time to expiration (years)
            n_std: Number of standard deviations to calculate

        Returns:
            Dictionary of standard deviation ranges
        """
        # Calculate period-adjusted standard deviation
        annual_std_dollars = volatility * current_price
        period_std = annual_std_dollars * np.sqrt(time_to_expiry)

        ranges = {}

        # Probabilities for normal distribution
        std_probabilities = {
            1: 0.6827,  # 68.27%
            2: 0.9545,  # 95.45%
            3: 0.9973   # 99.73%
        }

        for std in range(1, n_std + 1):
            lower = current_price - std * period_std
            upper = current_price + std * period_std

            ranges[f'{std}_sigma'] = {
                'lower_bound': max(0, float(lower)),
                'upper_bound': float(upper),
                'range_dollars': float(2 * std * period_std),
                'range_pct': float((2 * std * period_std) / current_price),
                'probability': std_probabilities.get(std, norm.cdf(std) - norm.cdf(-std))
            }

        return ranges

    def calculate_strike_probabilities(
        self,
        current_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float
    ) -> Dict:
        """
        Calculate probabilities related to a specific strike

        Args:
            current_price: Current stock price
            strike: Target strike price
            time_to_expiry: Time to expiration (years)
            volatility: Annualized volatility

        Returns:
            Dictionary of strike-related probabilities
        """
        # Lognormal distribution parameters
        log_S = np.log(current_price)
        log_K = np.log(strike)

        # Risk-neutral drift
        mean_log = log_S + (self.risk_free_rate - 0.5 * volatility**2) * time_to_expiry
        std_log = volatility * np.sqrt(time_to_expiry)

        # P(S_T > K) - probability of finishing in the money for a call
        prob_above = 1 - norm.cdf(log_K, mean_log, std_log)

        # P(S_T < K) - probability of finishing in the money for a put
        prob_below = norm.cdf(log_K, mean_log, std_log)

        # P(S_T touches K before expiration) - using barrier probability approximation
        prob_touch = self._calculate_touch_probability(
            current_price, strike, time_to_expiry, volatility
        )

        # Calculate z-score (how many std deviations away)
        expected_price = current_price * np.exp(self.risk_free_rate * time_to_expiry)
        period_std = volatility * current_price * np.sqrt(time_to_expiry)
        z_score = (strike - expected_price) / period_std if period_std > 0 else 0

        return {
            'prob_above_strike': float(prob_above),
            'prob_below_strike': float(prob_below),
            'prob_touch_strike': float(prob_touch),
            'expected_price_at_expiry': float(expected_price),
            'strike_distance_std': float(abs(z_score)),
            'strike_vs_current_pct': float((strike / current_price - 1))
        }

    def calculate_breakeven_probability(
        self,
        current_price: float,
        strike: float,
        option_premium: float,
        time_to_expiry: float,
        volatility: float,
        option_type: str = 'call'
    ) -> Dict:
        """
        Calculate probability of reaching breakeven for option position

        Args:
            current_price: Current stock price
            strike: Option strike
            option_premium: Option price paid
            time_to_expiry: Time to expiration (years)
            volatility: Annualized volatility
            option_type: 'call' or 'put'

        Returns:
            Breakeven analysis dictionary
        """
        # Calculate breakeven price
        if option_type == 'call':
            breakeven = strike + option_premium
            required_move = (breakeven / current_price - 1)
        else:  # put
            breakeven = strike - option_premium
            required_move = (breakeven / current_price - 1)

        # Probability of reaching breakeven
        log_S = np.log(current_price)
        log_BE = np.log(breakeven)
        mean_log = log_S + (self.risk_free_rate - 0.5 * volatility**2) * time_to_expiry
        std_log = volatility * np.sqrt(time_to_expiry)

        if option_type == 'call':
            prob_breakeven = 1 - norm.cdf(log_BE, mean_log, std_log)
        else:
            prob_breakeven = norm.cdf(log_BE, mean_log, std_log)

        # Calculate how many standard deviations away breakeven is
        expected_price = current_price * np.exp(self.risk_free_rate * time_to_expiry)
        period_std = volatility * current_price * np.sqrt(time_to_expiry)
        std_to_breakeven = abs(breakeven - expected_price) / period_std if period_std > 0 else 0

        return {
            'breakeven_price': float(breakeven),
            'required_move_pct': float(required_move),
            'required_move_dollars': float(breakeven - current_price),
            'probability_breakeven': float(prob_breakeven),
            'std_deviations_to_breakeven': float(std_to_breakeven),
            'option_premium': float(option_premium)
        }

    def monte_carlo_expected_value(
        self,
        current_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        option_premium: float,
        option_type: str = 'call',
        n_simulations: int = 10000
    ) -> Dict:
        """
        Monte Carlo simulation for expected value and P&L distribution

        Args:
            current_price: Current stock price
            strike: Option strike
            time_to_expiry: Time to expiration (years)
            volatility: Annualized volatility
            option_premium: Option price paid
            option_type: 'call' or 'put'
            n_simulations: Number of Monte Carlo paths

        Returns:
            Expected value analysis dictionary
        """
        # Simulate final prices using geometric Brownian motion
        dt = time_to_expiry
        drift = (self.risk_free_rate - 0.5 * volatility**2) * dt
        diffusion = volatility * np.sqrt(dt)

        z = np.random.standard_normal(n_simulations)
        final_prices = current_price * np.exp(drift + diffusion * z)

        # Calculate payoffs
        if option_type == 'call':
            intrinsic_values = np.maximum(final_prices - strike, 0)
        else:  # put
            intrinsic_values = np.maximum(strike - final_prices, 0)

        # Calculate P&L (payoff - premium paid)
        pnls = intrinsic_values - option_premium

        # Calculate statistics
        expected_value = np.mean(pnls)
        probability_profit = np.sum(pnls > 0) / n_simulations
        var_95 = np.percentile(pnls, 5)
        median_pnl = np.median(pnls)
        std_dev = np.std(pnls)

        # Max gain/loss
        max_gain = np.max(pnls)
        max_loss = np.min(pnls)

        # Calculate percentage returns
        pct_returns = pnls / option_premium if option_premium > 0 else pnls

        return {
            'expected_value': float(expected_value),
            'expected_return_pct': float(expected_value / option_premium * 100) if option_premium > 0 else 0,
            'probability_profit': float(probability_profit),
            'median_pnl': float(median_pnl),
            'std_dev': float(std_dev),
            'var_95': float(var_95),
            'max_gain': float(max_gain),
            'max_loss': float(max_loss),
            'max_loss_pct': float(max_loss / option_premium * 100) if option_premium > 0 else -100,
            'sharpe_ratio': float(expected_value / std_dev * np.sqrt(252 / (time_to_expiry * 365))) if std_dev > 0 else 0,
            'n_simulations': n_simulations
        }

    def calculate_target_probability(
        self,
        current_price: float,
        target_price: float,
        time_to_expiry: float,
        volatility: float,
        use_risk_neutral: bool = True
    ) -> float:
        """
        Calculate probability of reaching a target price

        Args:
            current_price: Current stock price
            target_price: Target price level
            time_to_expiry: Time horizon (years)
            volatility: Annualized volatility
            use_risk_neutral: Use risk-neutral (True) or historical drift (False)

        Returns:
            Probability of reaching target
        """
        log_S = np.log(current_price)
        log_target = np.log(target_price)

        if use_risk_neutral:
            drift = self.risk_free_rate
        else:
            drift = 0.08  # Assume 8% historical equity return

        mean_log = log_S + (drift - 0.5 * volatility**2) * time_to_expiry
        std_log = volatility * np.sqrt(time_to_expiry)

        if target_price > current_price:
            prob = 1 - norm.cdf(log_target, mean_log, std_log)
        else:
            prob = norm.cdf(log_target, mean_log, std_log)

        return float(prob)

    def _calculate_touch_probability(
        self,
        current_price: float,
        barrier: float,
        time_to_expiry: float,
        volatility: float
    ) -> float:
        """
        Calculate probability of touching a barrier before expiration
        Uses barrier option probability formula
        """
        if current_price == barrier:
            return 1.0

        # Barrier probability formula
        # P(touch) ≈ 2 * N(d) where d depends on barrier level
        s = current_price
        b = barrier
        t = time_to_expiry
        sigma = volatility

        if abs(np.log(b/s)) < 1e-10:
            return 1.0

        # Simplified barrier touch probability
        d = abs(np.log(b/s)) / (sigma * np.sqrt(t))
        prob_touch = 2 * (1 - norm.cdf(d))

        return min(1.0, float(prob_touch))
