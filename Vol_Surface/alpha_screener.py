"""
Alpha Screener

Identifies trading opportunities by analyzing:
1. IV vs HV spreads (volatility mispricing)
2. Vol surface arbitrage opportunities
3. Calendar spread anomalies
4. Butterfly arbitrage
5. Best-priced straddles and strangles
6. Term structure opportunities

Outputs ranked signals with confidence scores
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

from greeks_calculator import GreeksCalculator
from vol_arbitrage import VolatilityArbitrageStrategies

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlphaScreener:
    """
    Screens options for alpha-generating opportunities
    """

    def __init__(
        self,
        greeks_calc: GreeksCalculator,
        vol_arb: VolatilityArbitrageStrategies
    ):
        """
        Initialize screener

        Args:
            greeks_calc: GreeksCalculator instance
            vol_arb: VolatilityArbitrageStrategies instance
        """
        self.greeks = greeks_calc
        self.vol_arb = vol_arb

    def screen_opportunities(
        self,
        options_df: pd.DataFrame,
        current_price: float,
        historical_vol: float,
        ticker: str,
        seasonality_df: pd.DataFrame = None
    ) -> List[Dict]:
        """
        Comprehensive screening for all opportunity types

        Args:
            options_df: Options data
            current_price: Current underlying price
            historical_vol: Historical volatility
            ticker: Ticker symbol

        Returns:
            List of alpha signals, sorted by strength
        """
        logger.info(f"Screening {ticker} for opportunities...")

        all_signals = []

        # 1. IV vs HV Analysis
        iv_hv_signals = self._screen_iv_vs_hv(options_df, historical_vol, current_price)
        all_signals.extend(iv_hv_signals)

        # 2. Best Priced Straddles
        straddle_signals = self._screen_best_straddles(options_df, current_price, historical_vol)
        all_signals.extend(straddle_signals)

        # 3. Best Priced Strangles
        strangle_signals = self._screen_best_strangles(options_df, current_price, historical_vol)
        all_signals.extend(strangle_signals)

        # 4. Calendar Spread Arbitrage (seasonality-aware gating if available)
        calendar_signals = self._screen_calendar_spreads(options_df, current_price)
        if seasonality_df is not None and not seasonality_df.empty:
            calendar_signals = self._apply_seasonality_gate(calendar_signals, seasonality_df)
        all_signals.extend(calendar_signals)

        # 5. Butterfly Arbitrage
        butterfly_signals = self._screen_butterfly_spreads(options_df, current_price)
        all_signals.extend(butterfly_signals)

        # 6. Term Structure Analysis
        term_signals = self._screen_term_structure(options_df, current_price, historical_vol)
        all_signals.extend(term_signals)

        # 7. Diagonal Spreads (seasonality+term carry bias)
        diagonal_signals = self._screen_diagonal_spreads(options_df, current_price, seasonality_df)
        all_signals.extend(diagonal_signals)

        # Sort by strength score
        all_signals = sorted(all_signals, key=lambda x: x.get('strength_score', 0), reverse=True)

        logger.info(f"Found {len(all_signals)} total signals")

        return all_signals

    def _apply_seasonality_gate(self, signals: List[Dict], seasonality_df: pd.DataFrame) -> List[Dict]:
        gated = []
        if not signals:
            return gated
        if seasonality_df is None or seasonality_df.empty:
            return signals
        # Map expiry -> z
        zmap = {pd.Timestamp(r['expiration']).to_pydatetime().date(): r['z_seasonal'] for _, r in seasonality_df.iterrows()}
        for s in signals:
            exp = s.get('expiration') or s.get('long_expiry') or s.get('short_expiry')
            if exp is None:
                continue
            exp_date = pd.Timestamp(exp).date()
            z = zmap.get(exp_date, 0.0)
            # Only keep strong seasonality dislocations
            if abs(z) >= 1.0:
                s['seasonality_z'] = float(z)
                gated.append(s)
        return gated

    def _screen_diagonal_spreads(
        self,
        df: pd.DataFrame,
        current_price: float,
        seasonality_df: pd.DataFrame | None
    ) -> List[Dict]:
        """
        Screen for diagonal opportunities: sell front short-dated, buy longer-dated slightly OTM.
        Bias by seasonality (prefer short front when z>0, long back when term supports).
        """
        signals = []
        if df.empty:
            return signals
        expiries = sorted(df['expiration'].unique())
        if len(expiries) < 2:
            return signals
        # Simple: pair nearest and next-nearest expiries
        front = expiries[0]
        back = expiries[1]
        g_front = df[df['expiration'] == front].copy()
        g_back = df[df['expiration'] == back].copy()
        if g_front.empty or g_back.empty:
            return signals
        # ATM reference
        g_front['strike_diff'] = (g_front['strike'] - current_price).abs()
        atm_strike = g_front.loc[g_front['strike_diff'].idxmin(), 'strike']
        # Choose slight OTM for long leg
        otm_call = g_back[(g_back['option_type']=='call') & (g_back['strike'] >= current_price*1.02)]
        otm_put = g_back[(g_back['option_type']=='put') & (g_back['strike'] <= current_price*0.98)]
        # Build call diagonal if available
        if not otm_call.empty:
            long_strike = float(otm_call.iloc[0]['strike'])
            sig = {
                'type': 'DIAGONAL_SPREAD',
                'signal': 'CALL_DIAGONAL',
                'front_expiry': pd.Timestamp(front),
                'back_expiry': pd.Timestamp(back),
                'short_strike': float(atm_strike),
                'long_strike': long_strike,
                'recommendation': f'Sell {pd.Timestamp(front).date()} ATM call, Buy {pd.Timestamp(back).date()} {long_strike:.0f}C'
            }
            signals.append(sig)
        # Build put diagonal
        if not otm_put.empty:
            long_strike = float(otm_put.iloc[0]['strike'])
            sig = {
                'type': 'DIAGONAL_SPREAD',
                'signal': 'PUT_DIAGONAL',
                'front_expiry': pd.Timestamp(front),
                'back_expiry': pd.Timestamp(back),
                'short_strike': float(atm_strike),
                'long_strike': long_strike,
                'recommendation': f'Sell {pd.Timestamp(front).date()} ATM put, Buy {pd.Timestamp(back).date()} {long_strike:.0f}P'
            }
            signals.append(sig)
        # Apply seasonality bias: keep only when |z|>=1 if provided
        if seasonality_df is not None and not seasonality_df.empty:
            zmap = {pd.Timestamp(r['expiration']).to_pydatetime().date(): r['z_seasonal'] for _, r in seasonality_df.iterrows()}
            filtered = []
            for s in signals:
                z_front = zmap.get(pd.Timestamp(s['front_expiry']).date(), 0.0)
                if abs(z_front) >= 1.0:
                    s['seasonality_z'] = float(z_front)
                    filtered.append(s)
            signals = filtered
        return signals

    def _screen_iv_vs_hv(
        self,
        df: pd.DataFrame,
        historical_vol: float,
        current_price: float
    ) -> List[Dict]:
        """Screen for IV vs HV mispricing"""

        signals = []

        if df.empty or 'impliedVolatility' not in df.columns:
            return signals

        # Filter ATM options for cleaner IV reading
        df_atm = df[abs(df['strike'] - current_price) / current_price < 0.05].copy()

        if df_atm.empty:
            return signals

        # Group by expiration
        for expiry, group in df_atm.groupby('expiration'):
            avg_iv = group['impliedVolatility'].mean()

            if np.isnan(avg_iv) or avg_iv == 0:
                continue

            spread = avg_iv - historical_vol
            spread_pct = (spread / historical_vol) * 100

            # Significant mispricing threshold
            if abs(spread) > 0.05:  # 5% absolute difference

                signal_type = 'SELL_VOLATILITY' if spread > 0 else 'BUY_VOLATILITY'
                action = 'Short straddle' if spread > 0 else 'Long straddle'

                # Strength score (0-100)
                strength_score = min(100, abs(spread_pct) * 2)

                signals.append({
                    'type': 'IV_HV_SPREAD',
                    'signal': signal_type,
                    'action': action,
                    'expiration': expiry,
                    'implied_vol': avg_iv,
                    'historical_vol': historical_vol,
                    'spread': spread,
                    'spread_pct': spread_pct,
                    'strength_score': strength_score,
                    'confidence': 'HIGH' if abs(spread) > 0.10 else 'MEDIUM',
                    'recommendation': f"{action} at {expiry.strftime('%Y-%m-%d')}"
                })

        return signals

    def _screen_best_straddles(
        self,
        df: pd.DataFrame,
        current_price: float,
        historical_vol: float
    ) -> List[Dict]:
        """Find best-priced straddles relative to HV"""

        signals = []

        if df.empty:
            return signals

        # Find ATM straddles
        for expiry, group in df.groupby('expiration'):
            # Find closest strike to ATM
            group_sorted = group.copy()
            group_sorted['strike_diff'] = abs(group_sorted['strike'] - current_price)
            atm_strike = group_sorted.loc[group_sorted['strike_diff'].idxmin(), 'strike']

            # Get call and put at this strike
            call = group[(group['strike'] == atm_strike) & (group['option_type'] == 'call')]
            put = group[(group['strike'] == atm_strike) & (group['option_type'] == 'put')]

            if call.empty or put.empty:
                continue

            call_price = call.iloc[0]['mid'] if 'mid' in call.columns else call.iloc[0]['last']
            put_price = put.iloc[0]['mid'] if 'mid' in put.columns else put.iloc[0]['last']

            if pd.isna(call_price) or pd.isna(put_price) or call_price == 0 or put_price == 0:
                continue

            straddle_price = call_price + put_price

            # Get IV
            call_iv = call.iloc[0].get('impliedVolatility', 0)
            put_iv = put.iloc[0].get('impliedVolatility', 0)
            avg_iv = (call_iv + put_iv) / 2

            if avg_iv == 0:
                continue

            # Calculate days to expiration
            dte = (expiry - pd.Timestamp.now()).days

            if dte < 7:
                continue

            # Expected move (IV-implied)
            expected_move = current_price * avg_iv * np.sqrt(dte / 365)

            # Expected move (HV-implied)
            expected_move_hv = current_price * historical_vol * np.sqrt(dte / 365)

            # Straddle value relative to expected move
            straddle_to_move_ratio = straddle_price / expected_move if expected_move > 0 else 0

            # Determine if cheap or expensive
            iv_hv_spread = avg_iv - historical_vol

            if iv_hv_spread < -0.05:
                # IV < HV: Straddle is cheap
                signal = 'BUY_STRADDLE'
                strength_score = min(100, abs(iv_hv_spread) * 300)

                signals.append({
                    'type': 'CHEAP_STRADDLE',
                    'signal': signal,
                    'action': f'Long straddle {atm_strike}',
                    'expiration': expiry,
                    'strike': atm_strike,
                    'straddle_price': straddle_price,
                    'implied_vol': avg_iv,
                    'historical_vol': historical_vol,
                    'expected_move': expected_move,
                    'expected_move_hv': expected_move_hv,
                    'straddle_to_move_ratio': straddle_to_move_ratio,
                    'days_to_expiry': dte,
                    'strength_score': strength_score,
                    'confidence': 'HIGH' if abs(iv_hv_spread) > 0.10 else 'MEDIUM',
                    'recommendation': f'Buy {atm_strike} straddle exp {expiry.strftime("%Y-%m-%d")} @ ${straddle_price:.2f}'
                })

            elif iv_hv_spread > 0.05:
                # IV > HV: Straddle is expensive
                signal = 'SELL_STRADDLE'
                strength_score = min(100, iv_hv_spread * 300)

                signals.append({
                    'type': 'EXPENSIVE_STRADDLE',
                    'signal': signal,
                    'action': f'Short straddle {atm_strike}',
                    'expiration': expiry,
                    'strike': atm_strike,
                    'straddle_price': straddle_price,
                    'implied_vol': avg_iv,
                    'historical_vol': historical_vol,
                    'expected_move': expected_move,
                    'expected_move_hv': expected_move_hv,
                    'straddle_to_move_ratio': straddle_to_move_ratio,
                    'days_to_expiry': dte,
                    'strength_score': strength_score,
                    'confidence': 'HIGH' if iv_hv_spread > 0.10 else 'MEDIUM',
                    'recommendation': f'Sell {atm_strike} straddle exp {expiry.strftime("%Y-%m-%d")} @ ${straddle_price:.2f}'
                })

        return signals

    def _screen_best_strangles(
        self,
        df: pd.DataFrame,
        current_price: float,
        historical_vol: float
    ) -> List[Dict]:
        """Find best-priced strangles (OTM call + OTM put)"""

        signals = []

        if df.empty:
            return signals

        # For each expiration, find 10-delta strangle
        for expiry, group in df.groupby('expiration'):
            dte = (expiry - pd.Timestamp.now()).days
            if dte < 7:
                continue

            # Find OTM calls and puts
            otm_calls = group[(group['option_type'] == 'call') & (group['strike'] > current_price)]
            otm_puts = group[(group['option_type'] == 'put') & (group['strike'] < current_price)]

            if otm_calls.empty or otm_puts.empty:
                continue

            # Find strikes approximately 10% OTM
            target_call_strike = current_price * 1.10
            target_put_strike = current_price * 0.90

            # Find closest strikes
            otm_calls = otm_calls.copy()
            otm_puts = otm_puts.copy()
            otm_calls['strike_diff'] = abs(otm_calls['strike'] - target_call_strike)
            otm_puts['strike_diff'] = abs(otm_puts['strike'] - target_put_strike)

            call = otm_calls.loc[otm_calls['strike_diff'].idxmin()]
            put = otm_puts.loc[otm_puts['strike_diff'].idxmin()]

            call_price = call.get('mid', call.get('last', 0))
            put_price = put.get('mid', put.get('last', 0))

            if pd.isna(call_price) or pd.isna(put_price) or call_price == 0 or put_price == 0:
                continue

            strangle_price = call_price + put_price

            # Get IVs
            call_iv = call.get('impliedVolatility', 0)
            put_iv = put.get('impliedVolatility', 0)
            avg_iv = (call_iv + put_iv) / 2

            if avg_iv == 0:
                continue

            iv_hv_spread = avg_iv - historical_vol

            if abs(iv_hv_spread) > 0.05:
                signal_type = 'SELL_STRANGLE' if iv_hv_spread > 0 else 'BUY_STRANGLE'
                action = f"{'Short' if iv_hv_spread > 0 else 'Long'} {put['strike']}/{call['strike']} strangle"

                strength_score = min(100, abs(iv_hv_spread) * 250)

                signals.append({
                    'type': 'STRANGLE_OPPORTUNITY',
                    'signal': signal_type,
                    'action': action,
                    'expiration': expiry,
                    'put_strike': put['strike'],
                    'call_strike': call['strike'],
                    'strangle_price': strangle_price,
                    'implied_vol': avg_iv,
                    'historical_vol': historical_vol,
                    'iv_hv_spread': iv_hv_spread,
                    'days_to_expiry': dte,
                    'strength_score': strength_score,
                    'confidence': 'HIGH' if abs(iv_hv_spread) > 0.08 else 'MEDIUM',
                    'recommendation': f'{action} exp {expiry.strftime("%Y-%m-%d")} @ ${strangle_price:.2f}'
                })

        return signals

    def _screen_calendar_spreads(self, df: pd.DataFrame, current_price: float) -> List[Dict]:
        """Screen for calendar spread arbitrage"""

        signals = []

        if df.empty:
            return signals

        # Get unique expiries sorted
        expiries = sorted(df['expiration'].unique())

        if len(expiries) < 2:
            return signals

        # For each strike, check calendar spread across expiries
        for strike in df['strike'].unique():
            # Filter options at this strike
            strike_opts = df[df['strike'] == strike]

            # Get IVs by expiration
            ivs_by_exp = {}
            for exp in expiries:
                exp_opts = strike_opts[strike_opts['expiration'] == exp]
                if not exp_opts.empty:
                    avg_iv = exp_opts['impliedVolatility'].mean()
                    if not np.isnan(avg_iv) and avg_iv > 0:
                        dte = (exp - pd.Timestamp.now()).days
                        ivs_by_exp[exp] = {'iv': avg_iv, 'dte': dte}

            if len(ivs_by_exp) < 2:
                continue

            # Check for calendar arbitrage (variance should increase with maturity)
            sorted_exps = sorted(ivs_by_exp.keys())

            for i in range(len(sorted_exps) - 1):
                short_exp = sorted_exps[i]
                long_exp = sorted_exps[i + 1]

                short_iv = ivs_by_exp[short_exp]['iv']
                long_iv = ivs_by_exp[long_exp]['iv']

                short_dte = ivs_by_exp[short_exp]['dte']
                long_dte = ivs_by_exp[long_exp]['dte']

                # Total variance check
                short_var = short_iv ** 2 * (short_dte / 365)
                long_var = long_iv ** 2 * (long_dte / 365)

                # Arbitrage exists if long variance < short variance
                if long_var < short_var:
                    arbitrage_score = ((short_var - long_var) / short_var) * 100

                    if arbitrage_score > 5:  # Significant arbitrage
                        signals.append({
                            'type': 'CALENDAR_ARBITRAGE',
                            'signal': 'BUY_CALENDAR',
                            'action': f'Long {long_exp.strftime("%Y-%m-%d")}, Short {short_exp.strftime("%Y-%m-%d")}',
                            'strike': strike,
                            'short_expiry': short_exp,
                            'long_expiry': long_exp,
                            'short_iv': short_iv,
                            'long_iv': long_iv,
                            'short_var': short_var,
                            'long_var': long_var,
                            'arbitrage_score': arbitrage_score,
                            'strength_score': min(100, arbitrage_score * 2),
                            'confidence': 'HIGH',
                            'recommendation': f'Calendar spread at {strike}: Buy {long_dte}d, Sell {short_dte}d'
                        })

        return signals

    def _screen_butterfly_spreads(self, df: pd.DataFrame, current_price: float) -> List[Dict]:
        """Screen for butterfly arbitrage (convexity violations)"""

        signals = []

        if df.empty:
            return signals

        # For each expiration
        for expiry, group in df.groupby('expiration'):
            # Sort strikes
            strikes = sorted(group['strike'].unique())

            if len(strikes) < 3:
                continue

            # Check convexity for evenly-spaced strikes
            for i in range(len(strikes) - 2):
                k_low = strikes[i]
                k_mid = strikes[i + 1]
                k_high = strikes[i + 2]

                # Check if evenly spaced (within 10%)
                spacing1 = k_mid - k_low
                spacing2 = k_high - k_mid

                if abs(spacing1 - spacing2) / spacing1 > 0.10:
                    continue

                # Get IVs
                iv_low = group[group['strike'] == k_low]['impliedVolatility'].mean()
                iv_mid = group[group['strike'] == k_mid]['impliedVolatility'].mean()
                iv_high = group[group['strike'] == k_high]['impliedVolatility'].mean()

                if any(np.isnan([iv_low, iv_mid, iv_high])) or any([iv_low, iv_mid, iv_high]) == 0:
                    continue

                # Check convexity: mid IV should be <= average of low and high
                avg_iv = (iv_low + iv_high) / 2
                convexity_violation = iv_mid - avg_iv

                if convexity_violation > 0.02:  # Significant violation
                    arbitrage_score = convexity_violation * 500

                    signals.append({
                        'type': 'BUTTERFLY_ARBITRAGE',
                        'signal': 'BUY_BUTTERFLY',
                        'action': f'Butterfly {k_low}/{k_mid}/{k_high}',
                        'expiration': expiry,
                        'strike_low': k_low,
                        'strike_mid': k_mid,
                        'strike_high': k_high,
                        'iv_low': iv_low,
                        'iv_mid': iv_mid,
                        'iv_high': iv_high,
                        'convexity_violation': convexity_violation,
                        'strength_score': min(100, arbitrage_score),
                        'confidence': 'HIGH' if convexity_violation > 0.05 else 'MEDIUM',
                        'recommendation': f'Buy butterfly {k_low}/{k_mid}/{k_high} exp {expiry.strftime("%Y-%m-%d")}'
                    })

        return signals

    def _screen_term_structure(
        self,
        df: pd.DataFrame,
        current_price: float,
        historical_vol: float
    ) -> List[Dict]:
        """Screen for term structure anomalies"""

        signals = []

        if df.empty:
            return signals

        # Get ATM IVs across expiries
        expiries = sorted(df['expiration'].unique())

        if len(expiries) < 2:
            return signals

        iv_term_structure = []

        for exp in expiries:
            exp_group = df[df['expiration'] == exp]

            # Get ATM IV
            exp_group_copy = exp_group.copy()
            exp_group_copy['strike_diff'] = abs(exp_group_copy['strike'] - current_price)
            atm_opts = exp_group_copy[exp_group_copy['strike_diff'] == exp_group_copy['strike_diff'].min()]

            avg_iv = atm_opts['impliedVolatility'].mean()

            if not np.isnan(avg_iv) and avg_iv > 0:
                dte = (exp - pd.Timestamp.now()).days
                iv_term_structure.append({'expiry': exp, 'dte': dte, 'iv': avg_iv})

        if len(iv_term_structure) < 2:
            return signals

        # Analyze term structure shape
        dtes = [x['dte'] for x in iv_term_structure]
        ivs = [x['iv'] for x in iv_term_structure]

        # Calculate slope (using simple linear approximation)
        slope = (ivs[-1] - ivs[0]) / (dtes[-1] - dtes[0]) if dtes[-1] != dtes[0] else 0

        # Check for extreme slopes
        if slope > 0.001:  # Steep contango
            signals.append({
                'type': 'TERM_STRUCTURE',
                'signal': 'STEEP_CONTANGO',
                'action': 'Sell long-term vol, buy short-term vol',
                'slope': slope,
                'short_term_iv': ivs[0],
                'long_term_iv': ivs[-1],
                'historical_vol': historical_vol,
                'strength_score': min(100, slope * 50000),
                'confidence': 'MEDIUM',
                'recommendation': 'Sell calendar spread (fade contango)'
            })

        elif slope < -0.001:  # Steep backwardation
            signals.append({
                'type': 'TERM_STRUCTURE',
                'signal': 'STEEP_BACKWARDATION',
                'action': 'Buy long-term vol, sell short-term vol',
                'slope': slope,
                'short_term_iv': ivs[0],
                'long_term_iv': ivs[-1],
                'historical_vol': historical_vol,
                'strength_score': min(100, abs(slope) * 50000),
                'confidence': 'MEDIUM',
                'recommendation': 'Buy calendar spread (fade backwardation)'
            })

        # Check for short-term spike (mean reversion opportunity)
        if ivs[0] > ivs[-1] * 1.3 and ivs[0] > historical_vol * 1.5:
            signals.append({
                'type': 'VOL_SPIKE',
                'signal': 'SELL_SHORT_TERM_VOL',
                'action': 'Sell short-term volatility (mean reversion)',
                'short_term_iv': ivs[0],
                'long_term_iv': ivs[-1],
                'historical_vol': historical_vol,
                'spike_ratio': ivs[0] / ivs[-1],
                'strength_score': min(100, (ivs[0] / historical_vol - 1) * 100),
                'confidence': 'HIGH',
                'recommendation': 'Sell front-month straddle/strangle'
            })

        return signals


if __name__ == "__main__":
    # Test screener
    from data_manager import OptionsDataManager

    print("Testing Alpha Screener\n")

    data_mgr = OptionsDataManager()
    greeks = GreeksCalculator()
    vol_arb = VolatilityArbitrageStrategies(greeks)
    screener = AlphaScreener(greeks, vol_arb)

    # Test with AAPL
    ticker = 'AAPL'
    print(f"Fetching data for {ticker}...")

    options_df = data_mgr.fetch_options_data(ticker, 'yfinance')
    current_price = data_mgr.get_current_price(ticker)
    hv = data_mgr.calculate_historical_volatility(ticker)

    print(f"Current price: ${current_price:.2f}")
    print(f"Historical vol: {hv*100:.2f}%")
    print(f"Options available: {len(options_df)}\n")

    # Screen
    signals = screener.screen_opportunities(options_df, current_price, hv, ticker)

    print(f"\nFound {len(signals)} signals:\n")

    for i, signal in enumerate(signals[:10], 1):
        print(f"{i}. {signal['type']}")
        print(f"   Signal: {signal['signal']}")
        print(f"   Action: {signal['recommendation']}")
        print(f"   Strength: {signal['strength_score']:.0f}/100")
        print(f"   Confidence: {signal['confidence']}")
        print()

    print("Alpha Screener test complete!")
