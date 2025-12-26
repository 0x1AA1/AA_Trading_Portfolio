"""
Backtest Visualizer (Minimal)

Reads a CSV with columns like date, equity and plots an equity curve.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np


def _try_imports():
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception:
        go = None
    return pd, go


def plot_equity(csv_path: str, output_dir: str | None = None) -> str:
    pd, go = _try_imports()
    if pd is None:
        raise RuntimeError("pandas required for backtest visualization")
    df = pd.read_csv(csv_path)
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception:
            pass

    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / "data_cache" / "vol_surface")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir) / f"equity_curve_{ts}.html"

    if go is None:
        # fallback CSV copy
        out_csv = Path(output_dir) / f"equity_curve_{ts}.csv"
        df.to_csv(out_csv, index=False)
        return str(out_csv)

    x = df['date'] if 'date' in df.columns else np.arange(len(df))
    y = df['equity'] if 'equity' in df.columns else df.iloc[:, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Equity'))
    fig.update_layout(title='Equity Curve', xaxis_title='Date', yaxis_title='Equity')
    fig.write_html(out)
    return str(out)

