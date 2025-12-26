"""
Market Data Fetcher - Downloads market data (prices, volatility, indices)
Supports multiple data sources with fallback mechanisms.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os
import logging
from typing import List, Dict, Optional
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketDataFetcher:
    """Fetch and cache market data from various sources."""

    def __init__(self, cache_db_path: str):
        """Initialize market data fetcher with persistent cache."""
        self.cache_db = cache_db_path
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for caching market data."""
        os.makedirs(os.path.dirname(self.cache_db), exist_ok=True)
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                data_type TEXT,
                last_updated TEXT,
                PRIMARY KEY (symbol, date, data_type)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vix_data (
                date TEXT PRIMARY KEY,
                vix_close REAL,
                vix_high REAL,
                vix_low REAL,
                last_updated TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS baltic_dry_index (
                date TEXT PRIMARY KEY,
                bdi_value REAL,
                last_updated TEXT
            )
        ''')

        conn.commit()
        conn.close()
        logger.info(f"Market data database initialized at {self.cache_db}")

    def fetch_price_data(self,
                        symbols: List[str],
                        start_date: str,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch price data for multiple symbols.

        Parameters:
        -----------
        symbols : List[str]
            List of ticker symbols (e.g., ['GC=F', 'SI=F', '^GSPC'])
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : Optional[str]
            End date (defaults to current date)

        Returns:
        --------
        pd.DataFrame with multi-level columns (symbol, OHLCV)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        all_data = {}

        for symbol in symbols:
            # Check cache first
            cached = self._get_cached_market_data(symbol, start_date, end_date)

            if cached is not None and len(cached) > 0:
                logger.info(f"Retrieved {len(cached)} cached records for {symbol}")
                all_data[symbol] = cached
            else:
                # Fetch from yfinance
                try:
                    logger.info(f"Fetching {symbol} from yfinance")
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date)

                    if not df.empty:
                        df['symbol'] = symbol
                        self._cache_market_data(df, symbol)
                        all_data[symbol] = df
                    else:
                        logger.warning(f"No data retrieved for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol}: {e}")
                    continue

        if not all_data:
            logger.warning("No market data retrieved")
            return pd.DataFrame()

        # Combine all symbols
        combined = pd.concat(all_data, axis=1)
        return combined

    def _get_cached_market_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Retrieve market data from cache."""
        conn = sqlite3.connect(self.cache_db)

        query = '''
            SELECT date, open, high, low, close, volume
            FROM market_data
            WHERE symbol = ?
                AND date >= ?
                AND date <= ?
                AND data_type = 'price'
            ORDER BY date
        '''

        try:
            df = pd.read_sql_query(query, conn, params=[symbol, start_date, end_date])
            conn.close()

            if len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df.columns = [col.title() for col in df.columns]
                return df
            return None
        except Exception as e:
            conn.close()
            logger.error(f"Cache retrieval failed for {symbol}: {e}")
            return None

    def _cache_market_data(self, df: pd.DataFrame, symbol: str):
        """Store market data in cache."""
        if df.empty:
            return

        conn = sqlite3.connect(self.cache_db)

        # Prepare data for insertion
        cache_df = df.reset_index()
        cache_df.columns = cache_df.columns.str.lower()

        if 'date' not in cache_df.columns and 'index' in cache_df.columns:
            cache_df.rename(columns={'index': 'date'}, inplace=True)

        cache_df['symbol'] = symbol
        cache_df['data_type'] = 'price'
        cache_df['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Select relevant columns
        cache_df = cache_df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'data_type', 'last_updated']]

        # Insert or replace
        cache_df.to_sql('market_data', conn, if_exists='append', index=False)
        conn.commit()
        conn.close()

        logger.info(f"Cached {len(cache_df)} market data records for {symbol}")

    def fetch_vix_data(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch VIX (volatility index) data."""
        return self.fetch_price_data(['^VIX'], start_date, end_date)

    def fetch_baltic_dry_index(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch Baltic Dry Index data.
        Note: This may require a specialized data provider. Using synthetic data for demo.
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Generate synthetic BDI data
        np.random.seed(42)
        base = 1500
        values = []
        current = base

        for i in range(len(dates)):
            current *= (1 + np.random.normal(0.0005, 0.02))
            current += np.sin(2 * np.pi * i / 365) * 100
            values.append(current)

        df = pd.DataFrame({
            'date': dates,
            'BDI': values
        })

        df = df.set_index('date')
        logger.info(f"Generated {len(df)} Baltic Dry Index data points")

        return df

    def calculate_technical_indicators(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from price data."""
        indicators = pd.DataFrame(index=price_df.index)

        # Extract close prices (handle multi-level columns)
        if isinstance(price_df.columns, pd.MultiIndex):
            for symbol in price_df.columns.levels[0]:
                if 'Close' in price_df[symbol].columns:
                    close = price_df[symbol]['Close']

                    # Returns
                    indicators[f'{symbol}_return_1d'] = close.pct_change(1)
                    indicators[f'{symbol}_return_5d'] = close.pct_change(5)
                    indicators[f'{symbol}_return_20d'] = close.pct_change(20)

                    # Volatility
                    indicators[f'{symbol}_volatility_20d'] = close.pct_change().rolling(20).std() * np.sqrt(252)

                    # Moving averages
                    indicators[f'{symbol}_ma_20'] = close.rolling(20).mean()
                    indicators[f'{symbol}_ma_50'] = close.rolling(50).mean()

                    # RSI
                    delta = close.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    indicators[f'{symbol}_rsi'] = 100 - (100 / (1 + rs))

        else:
            # Single symbol or flat columns
            if 'Close' in price_df.columns:
                close = price_df['Close']

                indicators['return_1d'] = close.pct_change(1)
                indicators['return_5d'] = close.pct_change(5)
                indicators['return_20d'] = close.pct_change(20)
                indicators['volatility_20d'] = close.pct_change().rolling(20).std() * np.sqrt(252)
                indicators['ma_20'] = close.rolling(20).mean()
                indicators['ma_50'] = close.rolling(50).mean()

        return indicators

    def get_commodity_prices(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch key commodity prices.

        Commodities:
        - Gold (GC=F)
        - Silver (SI=F)
        - Oil WTI (CL=F)
        - Natural Gas (NG=F)
        - Copper (HG=F)
        """
        commodities = {
            'GOLD': 'GC=F',
            'SILVER': 'SI=F',
            'OIL_WTI': 'CL=F',
            'NAT_GAS': 'NG=F',
            'COPPER': 'HG=F'
        }

        symbols = list(commodities.values())
        df = self.fetch_price_data(symbols, start_date, end_date)

        return df


if __name__ == "__main__":
    # Test the market data fetcher
    cache_path = r"F:\1. perso - travail\2. Perso - Implementations\Coding\Python\Algo_Trade_1\Replicator\data\macro\market_data.db"

    fetcher = MarketDataFetcher(cache_path)

    # Fetch commodity prices
    try:
        commodities = fetcher.get_commodity_prices(start_date='2020-01-01', end_date='2025-09-30')
        print(f"\nCommodity data shape: {commodities.shape}")
        print(f"\nSample commodity data:\n{commodities.head()}")
    except Exception as e:
        print(f"Error fetching commodity data: {e}")

    # Fetch VIX
    try:
        vix = fetcher.fetch_vix_data(start_date='2020-01-01', end_date='2025-09-30')
        print(f"\nVIX data shape: {vix.shape}")
        print(f"\nSample VIX data:\n{vix.head()}")
    except Exception as e:
        print(f"Error fetching VIX: {e}")

    # Fetch Baltic Dry Index
    bdi = fetcher.fetch_baltic_dry_index(start_date='2020-01-01', end_date='2025-09-30')
    print(f"\nBaltic Dry Index shape: {bdi.shape}")
    print(f"\nSample BDI data:\n{bdi.head()}")
