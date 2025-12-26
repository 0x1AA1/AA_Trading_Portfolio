"""
Data collectors package for Trade Replicator.
Fetches data from multiple macro-economic sources.
"""

from .oecd_fetcher import OECDFetcher
from .market_data_fetcher import MarketDataFetcher

__all__ = ['OECDFetcher', 'MarketDataFetcher']
