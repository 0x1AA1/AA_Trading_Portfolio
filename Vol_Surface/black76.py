"""
Black 76 pricer for options on futures

Use for commodity futures options instead of Black-Scholes.
"""

from __future__ import annotations

import math
from math import log, sqrt, exp
from scipy.stats import norm  # type: ignore


def black76_price(F: float, K: float, T: float, sigma: float, r: float = 0.0, option_type: str = 'call') -> float:
    if T <= 0:
        return max(F - K, 0.0) if option_type == 'call' else max(K - F, 0.0)
    if sigma <= 0:
        return 0.0
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    df = math.exp(-r * T)
    if option_type == 'call':
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

