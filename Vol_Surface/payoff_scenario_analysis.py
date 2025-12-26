"""
Payoff and Scenario Analysis for Options Strategies

Features:
- Payoff diagrams for single and multi-leg strategies
- Monte Carlo scenario analysis
- Probability of profit calculations
- Interactive strategy builder
- Risk/reward metrics
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from scipy.integrate import quad
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class OptionLeg:
    """Represents a single option leg in a strategy"""

    def __init__(self, option_type, strike, premium, quantity, position):
        """
        Args:
            option_type: 'call' or 'put'
            strike: Strike price
            premium: Option premium paid/received
            quantity: Number of contracts
            position: 'long' or 'short'
        """
        self.option_type = option_type.lower()
        self.strike = strike
        self.premium = premium
        self.quantity = quantity
        self.position = position.lower()

    def payoff(self, spot_prices):
        """Calculate payoff at expiration"""
        spot_prices = np.array(spot_prices)

        if self.option_type == 'call':
            intrinsic = np.maximum(spot_prices - self.strike, 0)
        else:  # put
            intrinsic = np.maximum(self.strike - spot_prices, 0)

        if self.position == 'long':
            return self.quantity * (intrinsic - self.premium)
        else:  # short
            return self.quantity * (self.premium - intrinsic)

    def __repr__(self):
        return f"{self.position.upper()} {self.quantity}x {self.option_type.upper()} ${self.strike} @ ${self.premium}"


class PayoffAnalyzer:
    """Analyzes payoffs and scenarios for option strategies"""

    def __init__(self, current_price, risk_free_rate=0.05):
        self.current_price = current_price
        self.risk_free_rate = risk_free_rate
        self.legs = []

    def add_leg(self, option_type, strike, premium, quantity=1, position='long'):
        """Add option leg to strategy"""
        leg = OptionLeg(option_type, strike, premium, quantity, position)
        self.legs.append(leg)
        return leg

    def clear_legs(self):
        """Remove all legs"""
        self.legs = []

    def calculate_payoff(self, spot_prices):
        """Calculate total strategy payoff"""
        if not self.legs:
            return np.zeros_like(spot_prices)

        total_payoff = np.zeros_like(spot_prices, dtype=float)
        for leg in self.legs:
            total_payoff += leg.payoff(spot_prices)
        return total_payoff

    def get_breakevens(self, price_range=None):
        """Find breakeven points"""
        if price_range is None:
            strikes = [leg.strike for leg in self.legs]
            min_strike = min(strikes)
            max_strike = max(strikes)
            price_range = np.linspace(min_strike * 0.5, max_strike * 1.5, 1000)

        payoffs = self.calculate_payoff(price_range)

        # Find where payoff crosses zero
        breakevens = []
        for i in range(len(payoffs) - 1):
            if payoffs[i] * payoffs[i+1] < 0:  # Sign change
                # Linear interpolation
                be = price_range[i] + (price_range[i+1] - price_range[i]) * \
                     (-payoffs[i]) / (payoffs[i+1] - payoffs[i])
                breakevens.append(be)

        return breakevens

    def get_max_profit(self, price_range=None):
        """Calculate maximum profit"""
        if price_range is None:
            strikes = [leg.strike for leg in self.legs]
            min_strike = min(strikes)
            max_strike = max(strikes)
            price_range = np.linspace(min_strike * 0.5, max_strike * 1.5, 1000)

        payoffs = self.calculate_payoff(price_range)
        max_profit = np.max(payoffs)
        max_profit_price = price_range[np.argmax(payoffs)]

        return max_profit, max_profit_price

    def get_max_loss(self, price_range=None):
        """Calculate maximum loss"""
        if price_range is None:
            strikes = [leg.strike for leg in self.legs]
            min_strike = min(strikes)
            max_strike = max(strikes)
            price_range = np.linspace(min_strike * 0.5, max_strike * 1.5, 1000)

        payoffs = self.calculate_payoff(price_range)
        max_loss = np.min(payoffs)
        max_loss_price = price_range[np.argmin(payoffs)]

        return max_loss, max_loss_price

    def get_net_premium(self):
        """Calculate net premium paid/received"""
        net = 0
        for leg in self.legs:
            if leg.position == 'long':
                net -= leg.premium * leg.quantity
            else:
                net += leg.premium * leg.quantity
        return net

    def plot_payoff(self, title="Option Strategy Payoff", save_html=None):
        """Generate interactive payoff diagram"""
        if not self.legs:
            print("No legs added to strategy")
            return None

        # Price range
        strikes = [leg.strike for leg in self.legs]
        min_strike = min(strikes)
        max_strike = max(strikes)
        price_range = np.linspace(min_strike * 0.7, max_strike * 1.3, 500)

        # Calculate payoffs
        total_payoff = self.calculate_payoff(price_range)

        # Create figure
        fig = go.Figure()

        # Individual leg payoffs
        for i, leg in enumerate(self.legs):
            leg_payoff = leg.payoff(price_range)
            fig.add_trace(go.Scatter(
                x=price_range,
                y=leg_payoff,
                name=str(leg),
                line=dict(dash='dot', width=1),
                opacity=0.5
            ))

        # Total payoff
        fig.add_trace(go.Scatter(
            x=price_range,
            y=total_payoff,
            name='Total Payoff',
            line=dict(color='blue', width=3),
            fill='tozeroy',
            fillcolor='rgba(0,100,255,0.1)'
        ))

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        # Current price vertical line
        fig.add_vline(x=self.current_price, line_dash="dash",
                     line_color="green", opacity=0.7,
                     annotation_text=f"Current: ${self.current_price:.2f}")

        # Breakevens
        breakevens = self.get_breakevens(price_range)
        for be in breakevens:
            fig.add_vline(x=be, line_dash="dot", line_color="orange",
                         annotation_text=f"BE: ${be:.2f}")

        # Layout
        fig.update_layout(
            title=title,
            xaxis_title="Stock Price at Expiration",
            yaxis_title="Profit / Loss ($)",
            hovermode='x unified',
            template='plotly_dark',
            height=600,
            showlegend=True
        )

        if save_html:
            fig.write_html(save_html)

        return fig

    def monte_carlo_simulation(self, time_to_expiry, volatility, n_simulations=10000):
        """
        Run Monte Carlo simulation for probability analysis

        Args:
            time_to_expiry: Time to expiration in years
            volatility: Annualized volatility
            n_simulations: Number of price paths

        Returns:
            dict with simulation results
        """
        # Generate random price paths (Geometric Brownian Motion)
        dt = time_to_expiry
        drift = (self.risk_free_rate - 0.5 * volatility**2) * dt
        shock = volatility * np.sqrt(dt) * np.random.normal(0, 1, n_simulations)

        final_prices = self.current_price * np.exp(drift + shock)

        # Calculate payoffs
        payoffs = self.calculate_payoff(final_prices)

        # Statistics
        prob_profit = np.mean(payoffs > 0)
        avg_profit = np.mean(payoffs[payoffs > 0]) if np.any(payoffs > 0) else 0
        avg_loss = np.mean(payoffs[payoffs < 0]) if np.any(payoffs < 0) else 0
        expected_value = np.mean(payoffs)

        # Risk metrics
        var_95 = np.percentile(payoffs, 5)  # Value at Risk (95%)
        cvar_95 = np.mean(payoffs[payoffs <= var_95])  # Conditional VaR

        return {
            'final_prices': final_prices,
            'payoffs': payoffs,
            'prob_profit': prob_profit,
            'expected_value': expected_value,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'max_profit': np.max(payoffs),
            'max_loss': np.min(payoffs),
            'var_95': var_95,
            'cvar_95': cvar_95,
            'std_dev': np.std(payoffs)
        }

    def plot_monte_carlo(self, mc_results, title="Monte Carlo Payoff Distribution", save_html=None):
        """Plot Monte Carlo simulation results"""
        payoffs = mc_results['payoffs']

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Payoff Distribution', 'Price Distribution',
                          'Cumulative P&L', 'Risk Metrics'),
            specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]]
        )

        # Payoff distribution
        fig.add_trace(
            go.Histogram(x=payoffs, name='Payoffs', nbinsx=50,
                        marker_color='blue', opacity=0.7),
            row=1, col=1
        )

        # Price distribution
        fig.add_trace(
            go.Histogram(x=mc_results['final_prices'], name='Final Prices',
                        nbinsx=50, marker_color='green', opacity=0.7),
            row=1, col=2
        )

        # Cumulative P&L
        sorted_payoffs = np.sort(payoffs)
        cumulative_prob = np.arange(1, len(sorted_payoffs) + 1) / len(sorted_payoffs)
        fig.add_trace(
            go.Scatter(x=sorted_payoffs, y=cumulative_prob,
                      name='Cumulative Probability',
                      line=dict(color='purple', width=2)),
            row=2, col=1
        )

        # Risk metrics bar chart
        metrics = {
            'Expected Value': mc_results['expected_value'],
            'Avg Profit': mc_results['avg_profit'],
            'Avg Loss': mc_results['avg_loss'],
            'VaR 95%': mc_results['var_95'],
            'CVaR 95%': mc_results['cvar_95']
        }

        fig.add_trace(
            go.Bar(x=list(metrics.keys()), y=list(metrics.values()),
                  marker_color=['blue' if v >= 0 else 'red' for v in metrics.values()]),
            row=2, col=2
        )

        # Layout
        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=800,
            showlegend=False
        )

        fig.update_xaxes(title_text="Payoff ($)", row=1, col=1)
        fig.update_xaxes(title_text="Final Price ($)", row=1, col=2)
        fig.update_xaxes(title_text="Payoff ($)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)
        fig.update_yaxes(title_text="Value ($)", row=2, col=2)

        if save_html:
            fig.write_html(save_html)

        return fig

    def print_summary(self):
        """Print strategy summary"""
        print("=" * 80)
        print("STRATEGY SUMMARY")
        print("=" * 80)
        print(f"Current Price: ${self.current_price:.2f}")
        print(f"Number of Legs: {len(self.legs)}")
        print()

        for i, leg in enumerate(self.legs, 1):
            print(f"Leg {i}: {leg}")

        print()
        net_premium = self.get_net_premium()
        print(f"Net Premium: ${net_premium:.2f} ({'debit' if net_premium < 0 else 'credit'})")

        # Price range for analysis
        strikes = [leg.strike for leg in self.legs]
        min_strike = min(strikes)
        max_strike = max(strikes)
        price_range = np.linspace(min_strike * 0.5, max_strike * 1.5, 1000)

        max_profit, max_profit_price = self.get_max_profit(price_range)
        max_loss, max_loss_price = self.get_max_loss(price_range)
        breakevens = self.get_breakevens(price_range)

        print(f"\nMax Profit: ${max_profit:.2f} at ${max_profit_price:.2f}")
        print(f"Max Loss: ${max_loss:.2f} at ${max_loss_price:.2f}")

        if breakevens:
            print(f"\nBreakeven Points:")
            for i, be in enumerate(breakevens, 1):
                print(f"  BE{i}: ${be:.2f}")
        else:
            print("\nNo breakeven points found")

        print("=" * 80)


def create_common_strategy(strategy_name, current_price, atm_premium_call=None,
                          atm_premium_put=None, offset=5):
    """
    Create common option strategies

    Args:
        strategy_name: Strategy name (see list below)
        current_price: Current stock price
        atm_premium_call: ATM call premium
        atm_premium_put: ATM put premium
        offset: Strike offset for spreads

    Available strategies:
        - 'long_call', 'long_put', 'short_call', 'short_put'
        - 'long_straddle', 'short_straddle'
        - 'long_strangle', 'short_strangle'
        - 'bull_call_spread', 'bear_put_spread'
        - 'iron_condor', 'butterfly'
        - 'covered_call', 'protective_put'
    """
    analyzer = PayoffAnalyzer(current_price)
    atm_strike = round(current_price)

    # Use default premiums if not provided
    if atm_premium_call is None:
        atm_premium_call = current_price * 0.03  # 3% of stock price
    if atm_premium_put is None:
        atm_premium_put = current_price * 0.03

    strategy_name = strategy_name.lower()

    if strategy_name == 'long_call':
        analyzer.add_leg('call', atm_strike, atm_premium_call, 1, 'long')

    elif strategy_name == 'long_put':
        analyzer.add_leg('put', atm_strike, atm_premium_put, 1, 'long')

    elif strategy_name == 'short_call':
        analyzer.add_leg('call', atm_strike, atm_premium_call, 1, 'short')

    elif strategy_name == 'short_put':
        analyzer.add_leg('put', atm_strike, atm_premium_put, 1, 'short')

    elif strategy_name == 'long_straddle':
        analyzer.add_leg('call', atm_strike, atm_premium_call, 1, 'long')
        analyzer.add_leg('put', atm_strike, atm_premium_put, 1, 'long')

    elif strategy_name == 'short_straddle':
        analyzer.add_leg('call', atm_strike, atm_premium_call, 1, 'short')
        analyzer.add_leg('put', atm_strike, atm_premium_put, 1, 'short')

    elif strategy_name == 'long_strangle':
        analyzer.add_leg('call', atm_strike + offset, atm_premium_call * 0.7, 1, 'long')
        analyzer.add_leg('put', atm_strike - offset, atm_premium_put * 0.7, 1, 'long')

    elif strategy_name == 'short_strangle':
        analyzer.add_leg('call', atm_strike + offset, atm_premium_call * 0.7, 1, 'short')
        analyzer.add_leg('put', atm_strike - offset, atm_premium_put * 0.7, 1, 'short')

    elif strategy_name == 'bull_call_spread':
        analyzer.add_leg('call', atm_strike, atm_premium_call, 1, 'long')
        analyzer.add_leg('call', atm_strike + offset, atm_premium_call * 0.5, 1, 'short')

    elif strategy_name == 'bear_put_spread':
        analyzer.add_leg('put', atm_strike, atm_premium_put, 1, 'long')
        analyzer.add_leg('put', atm_strike - offset, atm_premium_put * 0.5, 1, 'short')

    elif strategy_name == 'iron_condor':
        analyzer.add_leg('put', atm_strike - offset * 2, atm_premium_put * 0.3, 1, 'long')
        analyzer.add_leg('put', atm_strike - offset, atm_premium_put * 0.6, 1, 'short')
        analyzer.add_leg('call', atm_strike + offset, atm_premium_call * 0.6, 1, 'short')
        analyzer.add_leg('call', atm_strike + offset * 2, atm_premium_call * 0.3, 1, 'long')

    elif strategy_name == 'butterfly':
        analyzer.add_leg('call', atm_strike - offset, atm_premium_call * 1.2, 1, 'long')
        analyzer.add_leg('call', atm_strike, atm_premium_call, 2, 'short')
        analyzer.add_leg('call', atm_strike + offset, atm_premium_call * 0.5, 1, 'long')

    elif strategy_name == 'covered_call':
        # Stock position
        analyzer.add_leg('call', atm_strike + offset, atm_premium_call * 0.7, 1, 'short')

    elif strategy_name == 'protective_put':
        # Stock position
        analyzer.add_leg('put', atm_strike - offset, atm_premium_put * 0.7, 1, 'long')

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return analyzer


class AdvancedProbabilityAnalyzer:
    """
    Advanced probability analysis for options using stochastic models

    Implements:
    - Girsanov theorem for measure changes (P to Q)
    - Heston stochastic volatility model
    - Barrier probabilities (touch/hit)
    - Historical vs Implied volatility comparison
    - Upside/downside potential with skewness

    References:
    - Girsanov (1960): Change of measure theorem
    - Heston (1993): Stochastic volatility model
    - Reiner-Rubinstein (1991): Barrier options
    - Bates (1996): Jumps and stochastic volatility
    """

    def __init__(self, current_price, risk_free_rate=0.05):
        self.S0 = current_price
        self.r = risk_free_rate

    def prob_itm_bs(self, strike, time_to_expiry, volatility, option_type='call'):
        """
        Probability of being in-the-money at expiration (Risk-neutral Q measure)

        Under Black-Scholes: P(ITM) = N(d2) for calls, N(-d2) for puts
        This uses the risk-neutral measure via Girsanov theorem

        Args:
            strike: Strike price
            time_to_expiry: Time to expiration (years)
            volatility: Annualized volatility
            option_type: 'call' or 'put'

        Returns:
            Probability of ITM at expiration under Q measure
        """
        if time_to_expiry <= 0:
            return 1.0 if (self.S0 > strike and option_type == 'call') or \
                         (self.S0 < strike and option_type == 'put') else 0.0

        d1 = (np.log(self.S0 / strike) + (self.r + 0.5 * volatility**2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        if option_type == 'call':
            return norm.cdf(d2)
        else:
            return norm.cdf(-d2)

    def prob_itm_physical(self, strike, time_to_expiry, volatility, drift, option_type='call'):
        """
        Probability of ITM under physical measure P (historical/realized)

        Uses actual expected return (drift) instead of risk-free rate
        This is the "real-world" probability using Girsanov change of measure

        Args:
            drift: Expected return under physical measure (e.g., historical return)

        Returns:
            Probability of ITM under P measure
        """
        if time_to_expiry <= 0:
            return 1.0 if (self.S0 > strike and option_type == 'call') or \
                         (self.S0 < strike and option_type == 'put') else 0.0

        # Under P measure, use drift instead of r
        d = (np.log(self.S0 / strike) + (drift - 0.5 * volatility**2) * time_to_expiry) / \
            (volatility * np.sqrt(time_to_expiry))

        if option_type == 'call':
            return norm.cdf(d)
        else:
            return norm.cdf(-d)

    def prob_touch_barrier(self, barrier, time_to_expiry, volatility):
        """
        Probability of touching a barrier before expiration

        Uses reflection principle and barrier option theory
        References: Merton (1973), Reiner-Rubinstein (1991)

        Args:
            barrier: Barrier level
            time_to_expiry: Time to expiration (years)
            volatility: Annualized volatility

        Returns:
            Probability of touching barrier before expiration
        """
        if time_to_expiry <= 0:
            return 1.0 if self.S0 >= barrier else 0.0

        if barrier <= self.S0:
            return 1.0  # Already above barrier

        # Formula: P(touch) = N(d) + (S0/H)^(2*mu/sigma^2) * N(d')
        # where mu = r - 0.5*sigma^2
        mu = self.r - 0.5 * volatility**2
        sigma_sqrt_t = volatility * np.sqrt(time_to_expiry)

        d = (np.log(self.S0 / barrier) + mu * time_to_expiry) / sigma_sqrt_t
        d_prime = (np.log(self.S0 / barrier) - mu * time_to_expiry) / sigma_sqrt_t

        # Reflection formula
        prob = norm.cdf(d) + (self.S0 / barrier)**(2 * mu / volatility**2) * norm.cdf(d_prime)

        return np.clip(prob, 0, 1)

    def heston_monte_carlo(self, time_to_expiry, v0, kappa, theta, sigma_v, rho, n_paths=10000, n_steps=100):
        """
        Heston stochastic volatility model via Monte Carlo

        dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW1_t
        dv_t = kappa * (theta - v_t) * dt + sigma_v * sqrt(v_t) * dW2_t
        where corr(dW1, dW2) = rho

        Args:
            v0: Initial variance (sigma^2)
            kappa: Mean reversion speed
            theta: Long-term variance
            sigma_v: Volatility of volatility
            rho: Correlation between asset and volatility
            n_paths: Number of simulation paths
            n_steps: Number of time steps

        Returns:
            dict with price paths and volatility paths
        """
        dt = time_to_expiry / n_steps

        # Initialize arrays
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = self.S0
        v[:, 0] = v0

        for i in range(n_steps):
            # Correlated Brownian motions
            dW1 = np.random.normal(0, np.sqrt(dt), n_paths)
            dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt), n_paths)

            # Update variance (use max to avoid negative variance)
            v[:, i+1] = np.maximum(
                v[:, i] + kappa * (theta - v[:, i]) * dt + sigma_v * np.sqrt(np.maximum(v[:, i], 0)) * dW2,
                0
            )

            # Update stock price
            S[:, i+1] = S[:, i] * np.exp(
                (self.r - 0.5 * v[:, i]) * dt + np.sqrt(np.maximum(v[:, i], 0)) * dW1
            )

        return {
            'price_paths': S,
            'volatility_paths': np.sqrt(v),
            'final_prices': S[:, -1],
            'final_volatilities': np.sqrt(v[:, -1])
        }

    def analyze_option_probabilities(self, strike, time_to_expiry, implied_vol, historical_vol,
                                    historical_drift, option_type='call'):
        """
        Comprehensive probability analysis comparing IV vs HV

        Args:
            strike: Option strike
            time_to_expiry: Time to expiration (years)
            implied_vol: Implied volatility from market
            historical_vol: Historical volatility
            historical_drift: Historical drift/return
            option_type: 'call' or 'put'

        Returns:
            dict with all probability metrics
        """
        # ITM probabilities
        prob_itm_iv = self.prob_itm_bs(strike, time_to_expiry, implied_vol, option_type)
        prob_itm_hv = self.prob_itm_physical(strike, time_to_expiry, historical_vol,
                                            historical_drift, option_type)

        # Touch probability (barrier at strike)
        prob_touch_iv = self.prob_touch_barrier(strike, time_to_expiry, implied_vol)
        prob_touch_hv = self.prob_touch_barrier(strike, time_to_expiry, historical_vol)

        # Expected moves
        expected_move_iv = self.S0 * implied_vol * np.sqrt(time_to_expiry)
        expected_move_hv = self.S0 * historical_vol * np.sqrt(time_to_expiry)

        # Upside/Downside analysis (1-sigma moves)
        upside_iv = self.S0 * np.exp((self.r - 0.5*implied_vol**2)*time_to_expiry + implied_vol*np.sqrt(time_to_expiry))
        downside_iv = self.S0 * np.exp((self.r - 0.5*implied_vol**2)*time_to_expiry - implied_vol*np.sqrt(time_to_expiry))

        upside_hv = self.S0 * np.exp((historical_drift - 0.5*historical_vol**2)*time_to_expiry + historical_vol*np.sqrt(time_to_expiry))
        downside_hv = self.S0 * np.exp((historical_drift - 0.5*historical_vol**2)*time_to_expiry - historical_vol*np.sqrt(time_to_expiry))

        # Volatility spread analysis
        vol_spread = implied_vol - historical_vol
        vol_spread_pct = (implied_vol / historical_vol - 1) * 100 if historical_vol > 0 else 0

        # Mispricing signal
        if vol_spread > 0.05:  # IV significantly higher than HV
            signal = "SELL_VOLATILITY"
            confidence = min(vol_spread * 10, 1.0)
        elif vol_spread < -0.05:  # HV significantly higher than IV
            signal = "BUY_VOLATILITY"
            confidence = min(abs(vol_spread) * 10, 1.0)
        else:
            signal = "NEUTRAL"
            confidence = 0.0

        return {
            # ITM Probabilities
            'prob_itm_implied': prob_itm_iv,
            'prob_itm_historical': prob_itm_hv,
            'prob_itm_diff': prob_itm_iv - prob_itm_hv,

            # Touch Probabilities
            'prob_touch_implied': prob_touch_iv,
            'prob_touch_historical': prob_touch_hv,
            'prob_touch_diff': prob_touch_iv - prob_touch_hv,

            # Expected Moves
            'expected_move_implied': expected_move_iv,
            'expected_move_historical': expected_move_hv,

            # Upside/Downside (1-sigma)
            'upside_implied': upside_iv,
            'downside_implied': downside_iv,
            'upside_historical': upside_hv,
            'downside_historical': downside_hv,

            # Potential analysis
            'upside_potential_pct': ((upside_iv / self.S0) - 1) * 100,
            'downside_risk_pct': ((downside_iv / self.S0) - 1) * 100,

            # Volatility spread
            'implied_vol': implied_vol,
            'historical_vol': historical_vol,
            'vol_spread': vol_spread,
            'vol_spread_pct': vol_spread_pct,

            # Trading signal
            'signal': signal,
            'confidence': confidence
        }

    def monte_carlo_barrier_analysis(self, strike, barriers, time_to_expiry, volatility,
                                    option_type='call', n_paths=10000, n_steps=252):
        """
        Monte Carlo simulation for complex barrier/path-dependent analysis

        Args:
            strike: Option strike
            barriers: dict with 'upper' and/or 'lower' barrier levels
            time_to_expiry: Time to expiration (years)
            volatility: Annualized volatility
            option_type: 'call' or 'put'
            n_paths: Number of paths
            n_steps: Number of time steps

        Returns:
            dict with barrier statistics
        """
        dt = time_to_expiry / n_steps
        sqrt_dt = np.sqrt(dt)

        # Initialize price paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0

        # Generate paths
        for i in range(n_steps):
            Z = np.random.normal(0, 1, n_paths)
            drift = (self.r - 0.5 * volatility**2) * dt
            diffusion = volatility * sqrt_dt * Z
            paths[:, i+1] = paths[:, i] * np.exp(drift + diffusion)

        # Barrier analysis
        results = {
            'paths': paths,
            'final_prices': paths[:, -1]
        }

        # Upper barrier analysis
        if 'upper' in barriers:
            upper = barriers['upper']
            touched_upper = np.any(paths >= upper, axis=1)
            results['prob_touch_upper'] = np.mean(touched_upper)
            results['avg_touch_time_upper'] = np.mean([
                np.argmax(paths[i] >= upper) * dt if touched_upper[i] else time_to_expiry
                for i in range(n_paths)
            ])

        # Lower barrier analysis
        if 'lower' in barriers:
            lower = barriers['lower']
            touched_lower = np.any(paths <= lower, axis=1)
            results['prob_touch_lower'] = np.mean(touched_lower)
            results['avg_touch_time_lower'] = np.mean([
                np.argmax(paths[i] <= lower) * dt if touched_lower[i] else time_to_expiry
                for i in range(n_paths)
            ])

        # Calculate payoffs
        final_prices = paths[:, -1]
        if option_type == 'call':
            payoffs = np.maximum(final_prices - strike, 0)
        else:
            payoffs = np.maximum(strike - final_prices, 0)

        results['payoffs'] = payoffs
        results['avg_payoff'] = np.mean(payoffs)
        results['prob_itm'] = np.mean(payoffs > 0)

        # Distribution moments
        results['price_mean'] = np.mean(final_prices)
        results['price_std'] = np.std(final_prices)
        results['price_skewness'] = skew(final_prices)
        results['price_kurtosis'] = kurtosis(final_prices)

        return results

    def print_probability_report(self, prob_analysis):
        """Print formatted probability analysis report"""
        print("\n" + "="*80)
        print("ADVANCED PROBABILITY ANALYSIS")
        print("="*80)

        print("\nITM PROBABILITIES (Risk-Neutral vs Physical Measure)")
        print("-" * 80)
        print(f"  P(ITM) - Implied Vol (Q-measure):     {prob_analysis['prob_itm_implied']:.2%}")
        print(f"  P(ITM) - Historical Vol (P-measure):  {prob_analysis['prob_itm_historical']:.2%}")
        print(f"  Difference (IV - HV):                 {prob_analysis['prob_itm_diff']:+.2%}")

        print("\nTOUCH PROBABILITIES (Barrier Analysis)")
        print("-" * 80)
        print(f"  P(Touch Strike) - Implied Vol:        {prob_analysis['prob_touch_implied']:.2%}")
        print(f"  P(Touch Strike) - Historical Vol:     {prob_analysis['prob_touch_historical']:.2%}")
        print(f"  Difference:                           {prob_analysis['prob_touch_diff']:+.2%}")

        print("\nEXPECTED MOVES (1-Sigma)")
        print("-" * 80)
        print(f"  Expected Move - Implied Vol:          ${prob_analysis['expected_move_implied']:.2f}")
        print(f"  Expected Move - Historical Vol:       ${prob_analysis['expected_move_historical']:.2f}")

        print("\nUPSIDE / DOWNSIDE POTENTIAL (1-Sigma Levels)")
        print("-" * 80)
        print(f"  Upside Target (IV):                   ${prob_analysis['upside_implied']:.2f} ({prob_analysis['upside_potential_pct']:+.2f}%)")
        print(f"  Downside Target (IV):                 ${prob_analysis['downside_implied']:.2f} ({prob_analysis['downside_risk_pct']:+.2f}%)")
        print(f"  Upside Target (HV):                   ${prob_analysis['upside_historical']:.2f}")
        print(f"  Downside Target (HV):                 ${prob_analysis['downside_historical']:.2f}")

        print("\nVOLATILITY SPREAD ANALYSIS")
        print("-" * 80)
        print(f"  Implied Volatility:                   {prob_analysis['implied_vol']:.2%}")
        print(f"  Historical Volatility:                {prob_analysis['historical_vol']:.2%}")
        print(f"  IV - HV Spread:                       {prob_analysis['vol_spread']:+.2%} ({prob_analysis['vol_spread_pct']:+.1f}%)")

        print("\nTRADING SIGNAL")
        print("-" * 80)
        print(f"  Signal:                               {prob_analysis['signal']}")
        print(f"  Confidence:                           {prob_analysis['confidence']:.1%}")

        if prob_analysis['signal'] == 'SELL_VOLATILITY':
            print(f"\n  Interpretation: IV > HV suggests options are overpriced.")
            print(f"                  Consider selling volatility (short straddle/strangle)")
        elif prob_analysis['signal'] == 'BUY_VOLATILITY':
            print(f"\n  Interpretation: HV > IV suggests options are underpriced.")
            print(f"                  Consider buying volatility (long straddle/strangle)")

        print("="*80)
