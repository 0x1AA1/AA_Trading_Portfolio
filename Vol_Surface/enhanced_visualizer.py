"""
Enhanced Visualizer

Minimal payoff diagram and Greeks grid visualizations using Plotly when available.
Outputs saved under data_cache/vol_surface with timestamps.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np


def _try_import_plotly():
    try:
        import plotly.graph_objects as go  # type: ignore
        return go
    except Exception:
        return None


def payoff_diagram(
    S0: float,
    positions: List[Dict],
    price_range_pct: float = 0.3,
    points: int = 200,
    output_dir: Path | None = None,
) -> str:
    """
    Plot payoff at expiry for a set of option positions.

    positions: list of dicts with keys:
      - type: 'call'|'put'
      - strike: float
      - qty: +1 for long, -1 for short
      - premium: price paid/received (optional, defaults 0)
    """
    go = _try_import_plotly()
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data_cache" / "vol_surface"
    output_dir.mkdir(parents=True, exist_ok=True)

    lo = max(1e-6, S0 * (1 - price_range_pct))
    hi = S0 * (1 + price_range_pct)
    S = np.linspace(lo, hi, points)

    payoff = np.zeros_like(S)
    for pos in positions:
        k = float(pos["strike"]) 
        qty = float(pos.get("qty", 1))
        prem = float(pos.get("premium", 0.0))
        if pos["type"].lower() == "call":
            payoff += qty * np.maximum(S - k, 0.0) - qty * prem
        else:
            payoff += qty * np.maximum(k - S, 0.0) - qty * prem

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_html = output_dir / f"payoff_{ts}.html"

    if go is None:
        # Fallback: save CSV
        out_csv = output_dir / f"payoff_{ts}.csv"
        data = np.vstack([S, payoff]).T
        np.savetxt(out_csv, data, delimiter=",", header="S,payoff", comments="")
        return str(out_csv)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S, y=payoff, mode="lines", name="Payoff"))
    fig.add_vline(x=S0, line_dash="dash", line_color="gray")
    fig.update_layout(title="Payoff Diagram", xaxis_title="Underlying Price", yaxis_title="P&L")
    fig.write_html(out_html)
    return str(out_html)

