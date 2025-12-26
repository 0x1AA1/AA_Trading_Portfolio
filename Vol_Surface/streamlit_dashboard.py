"""
Streamlit Dashboard (Minimal Scaffold)

Displays latest SVI parameter table and latest Monte Carlo summary if available.
"""

from __future__ import annotations

from pathlib import Path
import json
import glob


def _latest(pattern: str) -> str | None:
    files = glob.glob(pattern, recursive=True)
    return max(files) if files else None


def main():
    try:
        import streamlit as st  # type: ignore
        import pandas as pd  # type: ignore
    except Exception:
        print("Streamlit and pandas required. Install with: pip install streamlit pandas")
        return

    base = Path(__file__).parent.parent / "data_cache" / "vol_surface"
    st.title("Volatility Surface Dashboard (Minimal)")

    # Latest SVI params
    svi_path = _latest(str(base / "**" / "*_svi_params_*.csv"))
    if svi_path:
        st.subheader("Latest SVI Parameters")
        df = pd.read_csv(svi_path)
        st.write(Path(svi_path).name)
        st.dataframe(df)
    else:
        st.info("No SVI parameters found")

    # Latest MC summary
    mc_path = _latest(str(base / "**" / "*_mc_*.json"))
    if mc_path:
        st.subheader("Latest Monte Carlo Summary")
        st.write(Path(mc_path).name)
        with open(mc_path, 'r') as f:
            st.json(json.load(f))
    else:
        st.info("No Monte Carlo results found")


if __name__ == "__main__":
    main()
