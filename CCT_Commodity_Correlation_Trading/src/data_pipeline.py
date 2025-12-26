"""
Data Pipeline - Fetch and process commodity/equity price data
GPU/CPU accelerated where applicable
"""
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import logging

try:
    import cupy as cp
    GPU = True
except:
    GPU = False
    cp = np

logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu and GPU
        self.xp = cp if self.use_gpu else np
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
    def fetch_and_process(self, start, end):
        """Fetch commodity and equity data"""
        logger.info("Fetching price data...")
        
        # Default pairs
        pairs = [
            {"equity": "HL", "commodity": "SI=F", "name": "Silver-Hecla"},
            {"equity": "NEM", "commodity": "GC=F", "name": "Gold-Newmont"},
            {"equity": "FCX", "commodity": "HG=F", "name": "Copper-Freeport"},
            {"equity": "CVX", "commodity": "CL=F", "name": "Oil-Chevron"},
        ]
        
        tickers = list(set([p["equity"] for p in pairs] + [p["commodity"] for p in pairs]))
        
        # Download with auto_adjust=False to preserve Adj Close column
        raw_data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)

        # Extract Close prices (yfinance uses multi-level columns: (metric, ticker))
        if len(tickers) == 1:
            # Single ticker - columns are flat
            data = raw_data['Close'].to_frame(name=tickers[0])
        else:
            # Multi-ticker - columns are (metric, ticker)
            data = raw_data['Close']

        # Calculate returns
        returns = np.log(data / data.shift(1))
        
        logger.info(f"Downloaded {len(tickers)} tickers, {len(data)} days")
        
        return {"prices": data, "returns": returns, "pairs": pairs}
