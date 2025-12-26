"""
Decision Engine for Options Trading Recommendations

Synthesizes all analysis components into actionable BUY/HOLD/AVOID recommendations:
- Market context (price momentum, analyst targets)
- Volatility analysis (IV vs HV)
- Probability analysis (strike reach probability, expected value)
- Greeks and risk metrics

Provides scored recommendations with supporting rationale.

Author: Aoures ABDI
Version: 1.0
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Comprehensive decision engine for options trading recommendations
    """

    # Recommendation thresholds
    STRONG_BUY_THRESHOLD = 70
    BUY_THRESHOLD = 55
    HOLD_THRESHOLD = 45
    AVOID_THRESHOLD = 30

    def __init__(self):
        """Initialize the decision engine"""
        pass

    def evaluate_opportunity(
        self,
        market_context: Dict,
        volatility_analysis: Dict,
        probability_analysis: Dict,
        greeks: Optional[Dict] = None,
        strategy_metrics: Optional[Dict] = None
    ) -> Dict:
        """
        Comprehensive evaluation of an options opportunity

        Args:
            market_context: From MarketContextAnalyzer
            volatility_analysis: IV vs HV, volatility metrics
            probability_analysis: From ProbabilityAnalyzer
            greeks: Option Greeks (delta, gamma, vega, theta, rho)
            strategy_metrics: Monte Carlo strategy metrics

        Returns:
            Decision dictionary with recommendation and reasoning
        """
        logger.info("Evaluating options opportunity")

        # Initialize scoring
        score = 50  # Neutral baseline
        positive_factors = []
        negative_factors = []
        warnings = []

        # Weight distribution (total = 100 points possible)
        weights = {
            'valuation': 20,      # Price vs targets, valuations
            'timing': 20,         # Entry timing quality
            'probability': 25,    # Probability of success
            'volatility': 15,     # IV vs HV environment
            'risk_reward': 20     # Risk/reward profile
        }

        # 1. VALUATION ASSESSMENT (±20 points)
        valuation_score, val_factors = self._assess_valuation(market_context)
        score += valuation_score
        positive_factors.extend(val_factors['positive'])
        negative_factors.extend(val_factors['negative'])

        # 2. TIMING ASSESSMENT (±20 points)
        timing_score, timing_factors = self._assess_timing(market_context)
        score += timing_score
        positive_factors.extend(timing_factors['positive'])
        negative_factors.extend(timing_factors['negative'])
        warnings.extend(timing_factors.get('warnings', []))

        # 3. PROBABILITY ASSESSMENT (±25 points)
        prob_score, prob_factors = self._assess_probability(probability_analysis)
        score += prob_score
        positive_factors.extend(prob_factors['positive'])
        negative_factors.extend(prob_factors['negative'])

        # 4. VOLATILITY ENVIRONMENT (±15 points)
        vol_score, vol_factors = self._assess_volatility(volatility_analysis)
        score += vol_score
        positive_factors.extend(vol_factors['positive'])
        negative_factors.extend(vol_factors['negative'])

        # 5. RISK/REWARD ASSESSMENT (±20 points)
        if strategy_metrics or probability_analysis.get('expected_value_analysis'):
            rr_score, rr_factors = self._assess_risk_reward(
                probability_analysis.get('expected_value_analysis', {}),
                strategy_metrics or {}
            )
            score += rr_score
            positive_factors.extend(rr_factors['positive'])
            negative_factors.extend(rr_factors['negative'])

        # Ensure score stays in bounds
        final_score = max(0, min(100, score))

        # Generate recommendation
        recommendation = self._generate_recommendation(final_score)
        confidence = self._determine_confidence(final_score, len(positive_factors), len(negative_factors))

        # Calculate risk score (inverse of opportunity score)
        risk_score = 100 - final_score

        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'score': int(final_score),
            'risk_score': int(risk_score),
            'component_scores': {
                'valuation': int(50 + valuation_score),
                'timing': int(50 + timing_score),
                'probability': int(50 + prob_score),
                'volatility': int(50 + vol_score),
                'risk_reward': int(50 + rr_score) if 'rr_score' in locals() else 50
            },
            'positive_factors': positive_factors,
            'negative_factors': negative_factors,
            'warnings': warnings,
            'summary': self._generate_summary(
                final_score, recommendation, positive_factors, negative_factors
            )
        }

    def _assess_valuation(self, market_context: Dict) -> tuple:
        """Assess valuation and analyst sentiment"""
        score = 0
        factors = {'positive': [], 'negative': []}

        price_ctx = market_context.get('price_context', {})
        analyst_ctx = market_context.get('analyst_context', {})
        positioning = market_context.get('market_positioning', {})

        # Check price positioning
        percentile = price_ctx.get('percentile_in_52w_range', 0.5)
        if percentile > 0.95:
            score -= 10
            factors['negative'].append(f"Stock in top 5% of 52-week range (extreme high)")
        elif percentile > 0.85:
            score -= 5
            factors['negative'].append(f"Stock in top 15% of 52-week range (extended)")
        elif percentile < 0.15:
            score += 8
            factors['positive'].append(f"Stock in bottom 15% of 52-week range (potential value)")
        elif percentile < 0.30:
            score += 4
            factors['positive'].append(f"Stock below mid-range (reasonable entry)")

        # Check analyst targets
        upside_to_mean = analyst_ctx.get('upside_to_mean')
        if upside_to_mean is not None:
            if upside_to_mean > 0.25:
                score += 8
                factors['positive'].append(f"Significant upside to analyst mean target (+{upside_to_mean*100:.0f}%)")
            elif upside_to_mean > 0.10:
                score += 4
                factors['positive'].append(f"Moderate upside to analyst targets (+{upside_to_mean*100:.0f}%)")
            elif upside_to_mean < -0.10:
                score -= 8
                factors['negative'].append(f"Trading above analyst mean target ({upside_to_mean*100:.0f}% upside)")
            elif upside_to_mean < 0:
                score -= 4
                factors['negative'].append(f"At or above analyst targets (limited upside)")

        return score, factors

    def _assess_timing(self, market_context: Dict) -> tuple:
        """Assess entry timing quality"""
        score = 0
        factors = {'positive': [], 'negative': [], 'warnings': []}

        price_ctx = market_context.get('price_context', {})
        related_ctx = market_context.get('related_asset_context', {})

        # Check recent momentum
        one_month = price_ctx.get('1m_return', 0)
        three_month = price_ctx.get('3m_return', 0)

        if one_month > 0.20:
            score -= 10
            factors['negative'].append(f"Stock up {one_month*100:.0f}% in 1 month (potential chase)")
            factors['warnings'].append("WARNING: Buying after strong short-term rally - high FOMO risk")
        elif one_month > 0.10:
            score -= 5
            factors['negative'].append(f"Stock up {one_month*100:.0f}% in 1 month (extended)")
        elif one_month < -0.15:
            score += 8
            factors['positive'].append(f"Stock down {abs(one_month)*100:.0f}% in 1 month (potential reversal)")
        elif one_month < -0.05:
            score += 4
            factors['positive'].append(f"Recent pullback provides better entry")

        # Check momentum score
        momentum = price_ctx.get('momentum_score', 50)
        if momentum > 85:
            score -= 6
            factors['negative'].append(f"Very high momentum ({momentum}/100) - exhaustion risk")
        elif momentum < 20:
            score += 6
            factors['positive'].append(f"Low momentum ({momentum}/100) - potential reversal setup")

        # Check related asset if available
        related_percentile = related_ctx.get('percentile_in_3m_range')
        if related_percentile is not None:
            if related_percentile > 0.90:
                score -= 4
                factors['negative'].append("Related asset near 3-month highs - limited tailwind")
            elif related_percentile < 0.20:
                score += 4
                factors['positive'].append("Related asset near lows - potential support if recovers")

        return score, factors

    def _assess_probability(self, probability_analysis: Dict) -> tuple:
        """Assess probability of success"""
        score = 0
        factors = {'positive': [], 'negative': []}

        strike_probs = probability_analysis.get('strike_probabilities', {})
        breakeven = probability_analysis.get('breakeven_analysis', {})
        std_ranges = probability_analysis.get('std_ranges', {})

        # Probability of reaching strike
        prob_above = strike_probs.get('prob_above_strike', 0)
        if prob_above > 0.50:
            score += 12
            factors['positive'].append(f"High probability of reaching strike ({prob_above*100:.0f}%)")
        elif prob_above > 0.35:
            score += 6
            factors['positive'].append(f"Reasonable strike probability ({prob_above*100:.0f}%)")
        elif prob_above < 0.15:
            score -= 12
            factors['negative'].append(f"Low probability of reaching strike ({prob_above*100:.0f}%)")
        elif prob_above < 0.25:
            score -= 6
            factors['negative'].append(f"Below-average strike probability ({prob_above*100:.0f}%)")

        # Probability of profit (breakeven)
        if breakeven:
            prob_profit = breakeven.get('probability_breakeven', 0)
            if prob_profit > 0.50:
                score += 8
                factors['positive'].append(f"Favorable breakeven probability ({prob_profit*100:.0f}%)")
            elif prob_profit < 0.25:
                score -= 8
                factors['negative'].append(f"Low breakeven probability ({prob_profit*100:.0f}%)")

            # Standard deviations to breakeven
            std_to_be = breakeven.get('std_deviations_to_breakeven', 0)
            if std_to_be > 2.5:
                score -= 5
                factors['negative'].append(f"Breakeven {std_to_be:.1f} std dev away (unlikely)")

        return score, factors

    def _assess_volatility(self, volatility_analysis: Dict) -> tuple:
        """Assess volatility environment"""
        score = 0
        factors = {'positive': [], 'negative': []}

        iv_hv_spread = volatility_analysis.get('iv_hv_spread', 0)
        iv = volatility_analysis.get('implied_volatility', 0)
        hv = volatility_analysis.get('historical_volatility', 0)

        # For long options, want cheap IV (IV < HV)
        # For short options, want expensive IV (IV > HV)
        # Assume analyzing long option by default

        if iv_hv_spread < -0.10:
            score += 10
            factors['positive'].append(f"IV significantly below HV ({iv_hv_spread*100:.0f}% spread) - cheap options")
        elif iv_hv_spread < -0.05:
            score += 5
            factors['positive'].append(f"IV below HV - favorable for buying options")
        elif iv_hv_spread > 0.10:
            score -= 10
            factors['negative'].append(f"IV significantly above HV ({iv_hv_spread*100:+.0f}% spread) - expensive options")
        elif iv_hv_spread > 0.05:
            score -= 5
            factors['negative'].append(f"IV above HV - options expensive")

        return score, factors

    def _assess_risk_reward(self, ev_analysis: Dict, strategy_metrics: Dict) -> tuple:
        """Assess risk/reward profile"""
        score = 0
        factors = {'positive': [], 'negative': []}

        # Expected value
        expected_value = ev_analysis.get('expected_value') or strategy_metrics.get('expected_value', 0)
        expected_return = ev_analysis.get('expected_return_pct') or strategy_metrics.get('expected_return', 0)

        if expected_return > 30:
            score += 12
            factors['positive'].append(f"Strong expected return (+{expected_return:.0f}%)")
        elif expected_return > 10:
            score += 6
            factors['positive'].append(f"Positive expected return (+{expected_return:.0f}%)")
        elif expected_return < -10:
            score -= 10
            factors['negative'].append(f"Negative expected return ({expected_return:.0f}%)")

        # Probability of profit
        prob_profit = ev_analysis.get('probability_profit') or strategy_metrics.get('probability_profit', 0)
        if prob_profit > 0.60:
            score += 6
            factors['positive'].append(f"High win probability ({prob_profit*100:.0f}%)")
        elif prob_profit < 0.35:
            score -= 6
            factors['negative'].append(f"Low win probability ({prob_profit*100:.0f}%)")

        # Sharpe ratio
        sharpe = ev_analysis.get('sharpe_ratio') or strategy_metrics.get('sharpe', 0)
        if sharpe > 1.0:
            score += 2
            factors['positive'].append(f"Strong risk-adjusted return (Sharpe {sharpe:.2f})")
        elif sharpe < -0.5:
            score -= 2
            factors['negative'].append(f"Poor risk-adjusted return (Sharpe {sharpe:.2f})")

        return score, factors

    def _generate_recommendation(self, score: float) -> str:
        """Generate recommendation based on score"""
        if score >= self.STRONG_BUY_THRESHOLD:
            return 'STRONG BUY'
        elif score >= self.BUY_THRESHOLD:
            return 'BUY'
        elif score >= self.HOLD_THRESHOLD:
            return 'HOLD'
        elif score >= self.AVOID_THRESHOLD:
            return 'AVOID'
        else:
            return 'STRONG AVOID'

    def _determine_confidence(
        self,
        score: float,
        n_positive: int,
        n_negative: int
    ) -> str:
        """Determine confidence level in recommendation"""
        # Strong conviction if score is extreme and factors align
        if score > 75 or score < 25:
            return 'HIGH'
        elif score > 60 or score < 40:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _generate_summary(
        self,
        score: float,
        recommendation: str,
        positive_factors: List[str],
        negative_factors: List[str]
    ) -> str:
        """Generate executive summary of recommendation"""
        summary_lines = []

        summary_lines.append(f"Recommendation: {recommendation} (Score: {score:.0f}/100)")
        summary_lines.append("")

        if positive_factors:
            summary_lines.append(f"Positive Factors ({len(positive_factors)}):")
            for factor in positive_factors[:5]:  # Top 5
                summary_lines.append(f"  + {factor}")
            summary_lines.append("")

        if negative_factors:
            summary_lines.append(f"Negative Factors ({len(negative_factors)}):")
            for factor in negative_factors[:5]:  # Top 5
                summary_lines.append(f"  - {factor}")
            summary_lines.append("")

        # Overall assessment
        if score >= 70:
            summary_lines.append("Overall: Strong opportunity with favorable risk/reward")
        elif score >= 55:
            summary_lines.append("Overall: Reasonable opportunity worth considering")
        elif score >= 45:
            summary_lines.append("Overall: Neutral - wait for better setup")
        elif score >= 30:
            summary_lines.append("Overall: Unfavorable - multiple concerns")
        else:
            summary_lines.append("Overall: Poor opportunity - significant risks")

        return "\n".join(summary_lines)
