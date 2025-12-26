"""
QuantLib Validation Utilities (optional)

Provides simple price validation by comparing QuantLib European option
prices to Black–Scholes prices from GreeksCalculator.

If QuantLib is not installed, functions return status 'unavailable'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

try:
    import QuantLib as ql  # type: ignore
except Exception:
    ql = None  # type: ignore

from greeks_calculator import GreeksCalculator


@dataclass
class ValidationResult:
    status: str
    model_price: float
    quantlib_price: float
    abs_diff: float
    rel_diff_pct: float


def validate_european_price(S: float, K: float, T: float, sigma: float, r: float = 0.0, q: float = 0.0, option_type: str = 'call') -> Dict:
    """
    Compare Black–Scholes price to QuantLib price for a European option.

    Returns a dict with status and differences. If QuantLib is unavailable,
    status is 'unavailable'.
    """
    gc = GreeksCalculator(risk_free_rate=r, dividend_yield=q)
    bs_price = float(gc.black_scholes_price(S, K, T, sigma, option_type))

    if ql is None:
        return {
            'status': 'unavailable',
            'model': 'BlackScholes',
            'model_price': bs_price,
        }

    # QuantLib setup
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    settlement = ql.Settings.instance()
    # Anchor to evaluation date today
    eval_date = ql.Date.todaysDate()
    settlement.evaluationDate = eval_date

    maturity_date = eval_date + int(T * 365 + 0.5)

    payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put, K)
    exercise = ql.EuropeanExercise(maturity_date)

    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    risk_free = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, r, day_count))
    dividend = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, q, day_count))
    vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_date, calendar, sigma, day_count))

    process = ql.BlackScholesMertonProcess(spot, dividend, risk_free, vol)
    engine = ql.AnalyticEuropeanEngine(process)
    option = ql.EuropeanOption(payoff, exercise)
    option.setPricingEngine(engine)

    ql_price = float(option.NPV())

    abs_diff = abs(ql_price - bs_price)
    rel_diff_pct = abs_diff / max(1e-8, abs(bs_price)) * 100.0

    return {
        'status': 'ok',
        'model': 'BlackScholes_vs_QuantLib',
        'model_price': bs_price,
        'quantlib_price': ql_price,
        'abs_diff': abs_diff,
        'rel_diff_pct': rel_diff_pct,
    }

