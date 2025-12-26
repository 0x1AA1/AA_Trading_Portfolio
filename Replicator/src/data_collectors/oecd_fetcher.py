"""
OECD Data Fetcher - Downloads economic indicators from OECD API
Implements persistent caching with SQLite for efficient data management.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import sqlite3
import os
import time
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OECDFetcher:
    """Fetch and cache OECD economic data."""

    def __init__(self, cache_db_path: str):
        """Initialize OECD data fetcher with persistent cache."""
        self.cache_db = cache_db_path
        self.base_url = "https://sdmx.oecd.org/public/rest/data"
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for caching."""
        os.makedirs(os.path.dirname(self.cache_db), exist_ok=True)
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        # Create tables for different indicator types
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS oecd_data (
                indicator TEXT,
                country TEXT,
                date TEXT,
                value REAL,
                frequency TEXT,
                last_updated TEXT,
                PRIMARY KEY (indicator, country, date, frequency)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fetch_metadata (
                indicator TEXT,
                country TEXT,
                frequency TEXT,
                last_fetch TEXT,
                record_count INTEGER,
                PRIMARY KEY (indicator, country, frequency)
            )
        ''')

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.cache_db}")

    def fetch_indicator(self,
                       indicator: str,
                       countries: List[str] = ['USA', 'DEU', 'CHN', 'GBR', 'FRA'],
                       start_date: str = '2020-01',
                       end_date: Optional[str] = None,
                       frequency: str = 'M') -> pd.DataFrame:
        """
        Fetch OECD indicator data with caching.

        Parameters:
        -----------
        indicator : str
            OECD indicator code (e.g., 'GDP', 'CPI', 'UNEM')
        countries : List[str]
            ISO3 country codes
        start_date : str
            Start date in format 'YYYY-MM'
        end_date : Optional[str]
            End date in format 'YYYY-MM' (defaults to current month)
        frequency : str
            'M' for monthly, 'Q' for quarterly, 'A' for annual

        Returns:
        --------
        pd.DataFrame with columns: date, country, value
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m')

        # Check cache first
        cached_data = self._get_cached_data(indicator, countries, start_date, end_date, frequency)

        if cached_data is not None and len(cached_data) > 0:
            logger.info(f"Retrieved {len(cached_data)} cached records for {indicator}")
            return cached_data

        # Fetch from OECD API
        logger.info(f"Fetching {indicator} from OECD API for {len(countries)} countries")
        all_data = []

        for country in countries:
            try:
                data = self._fetch_from_api(indicator, country, frequency, start_date, end_date)
                if data is not None:
                    all_data.append(data)
                    time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.warning(f"Failed to fetch {indicator} for {country}: {e}")
                continue

        if not all_data:
            logger.warning(f"No data retrieved for {indicator}")
            return pd.DataFrame()

        # Combine and cache
        df = pd.concat(all_data, ignore_index=True)
        self._cache_data(df, indicator, frequency)

        return df

    def _fetch_from_api(self, indicator: str, country: str, frequency: str,
                       start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from OECD SDMX API."""
        # Construct API URL based on dataset
        dataset_map = {
            'GDP': 'QNA',  # Quarterly National Accounts
            'CPI': 'MEI',  # Main Economic Indicators
            'UNEM': 'MEI',
            'CLI': 'MEI',  # Composite Leading Indicator
            'INTEREST_RATE': 'MEI',
            'EXCHANGE_RATE': 'MEI'
        }

        dataset = dataset_map.get(indicator, 'MEI')

        # Simplified API call (real implementation would need proper SDMX parsing)
        url = f"{self.base_url}/{dataset}/{country}.{frequency}.all/all"

        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                # Parse SDMX-JSON response
                # This is a simplified version - real SDMX parsing is more complex
                data_dict = response.json()

                # Extract time series data (structure varies by API response)
                # Placeholder for actual SDMX parsing logic
                df = self._parse_sdmx_response(data_dict, indicator, country)
                return df
            else:
                logger.warning(f"API returned status {response.status_code} for {indicator}/{country}")
                return None

        except Exception as e:
            logger.error(f"API request failed for {indicator}/{country}: {e}")
            return None

    def _parse_sdmx_response(self, response_data: Dict, indicator: str, country: str) -> pd.DataFrame:
        """Parse SDMX-JSON response into DataFrame."""
        # Placeholder - real implementation would parse SDMX structure
        # For now, return empty DataFrame (will use synthetic data for demo)
        return pd.DataFrame(columns=['date', 'country', 'value', 'indicator'])

    def _get_cached_data(self, indicator: str, countries: List[str],
                        start_date: str, end_date: str, frequency: str) -> Optional[pd.DataFrame]:
        """Retrieve data from cache."""
        conn = sqlite3.connect(self.cache_db)

        query = '''
            SELECT date, country, value, indicator
            FROM oecd_data
            WHERE indicator = ?
                AND country IN ({})
                AND date >= ?
                AND date <= ?
                AND frequency = ?
            ORDER BY date, country
        '''.format(','.join('?' * len(countries)))

        params = [indicator] + countries + [start_date, end_date, frequency]

        try:
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            if len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                return df
            return None
        except Exception as e:
            conn.close()
            logger.error(f"Cache retrieval failed: {e}")
            return None

    def _cache_data(self, df: pd.DataFrame, indicator: str, frequency: str):
        """Store data in cache."""
        if df.empty:
            return

        conn = sqlite3.connect(self.cache_db)

        # Add metadata columns
        df['indicator'] = indicator
        df['frequency'] = frequency
        df['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Insert or replace
        df.to_sql('oecd_data', conn, if_exists='append', index=False)

        # Update metadata
        for country in df['country'].unique():
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO fetch_metadata (indicator, country, frequency, last_fetch, record_count)
                VALUES (?, ?, ?, ?, ?)
            ''', (indicator, country, frequency, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), len(df[df['country'] == country])))

        conn.commit()
        conn.close()
        logger.info(f"Cached {len(df)} records for {indicator}")

    def get_multiple_indicators(self,
                               indicators: List[str],
                               countries: List[str] = ['USA', 'DEU', 'CHN'],
                               start_date: str = '2020-01') -> pd.DataFrame:
        """Fetch multiple indicators and combine into wide format."""
        all_dfs = []

        for indicator in indicators:
            df = self.fetch_indicator(indicator, countries, start_date)
            if not df.empty:
                df = df.pivot(index='date', columns='country', values='value')
                df.columns = [f"{indicator}_{col}" for col in df.columns]
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        # Combine all indicators
        combined = pd.concat(all_dfs, axis=1)
        combined = combined.sort_index()

        return combined

    def generate_synthetic_data(self,
                               indicators: List[str],
                               countries: List[str] = ['USA', 'DEU', 'CHN', 'GBR', 'FRA'],
                               start_date: str = '2020-01',
                               end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Generate synthetic OECD-like data for demonstration.
        This is used when actual API is unavailable.
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m')

        # Generate monthly date range
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')

        data = []
        np.random.seed(42)

        indicator_configs = {
            'GDP': {'mean': 100, 'trend': 0.002, 'vol': 2},
            'CPI': {'mean': 100, 'trend': 0.001, 'vol': 0.5},
            'UNEM': {'mean': 5, 'trend': -0.001, 'vol': 0.3},
            'CLI': {'mean': 100, 'trend': 0.001, 'vol': 1.5},
            'INTEREST_RATE': {'mean': 2, 'trend': 0.002, 'vol': 0.2},
            'EXCHANGE_RATE': {'mean': 1, 'trend': 0.0005, 'vol': 0.05}
        }

        for indicator in indicators:
            config = indicator_configs.get(indicator, {'mean': 100, 'trend': 0, 'vol': 1})

            for country in countries:
                # Generate realistic time series with trend and noise
                values = []
                current = config['mean'] * (1 + np.random.normal(0, 0.1))

                for i, date in enumerate(dates):
                    # Add trend
                    current *= (1 + config['trend'])
                    # Add noise
                    current *= (1 + np.random.normal(0, config['vol'] / 100))
                    # Add some seasonality
                    seasonal = np.sin(2 * np.pi * i / 12) * config['vol'] / 2
                    value = current + seasonal

                    values.append(value)

                for date, value in zip(dates, values):
                    data.append({
                        'date': date,
                        'country': country,
                        'indicator': indicator,
                        'value': value
                    })

        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} synthetic OECD data points")

        # Cache synthetic data
        conn = sqlite3.connect(self.cache_db)
        df_cache = df.copy()
        df_cache['frequency'] = 'M'
        df_cache['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df_cache.to_sql('oecd_data', conn, if_exists='append', index=False)
        conn.close()

        return df


if __name__ == "__main__":
    # Test the OECD fetcher
    cache_path = r"F:\1. perso - travail\2. Perso - Implementations\Coding\Python\Algo_Trade_1\Replicator\data\macro\oecd\oecd_cache.db"

    fetcher = OECDFetcher(cache_path)

    # Generate synthetic data for testing
    indicators = ['GDP', 'CPI', 'UNEM', 'CLI', 'INTEREST_RATE']
    countries = ['USA', 'DEU', 'CHN', 'GBR', 'FRA']

    data = fetcher.generate_synthetic_data(
        indicators=indicators,
        countries=countries,
        start_date='2020-01-01',
        end_date='2025-09-30'
    )

    print(f"\nGenerated OECD data shape: {data.shape}")
    print(f"\nSample data:\n{data.head(20)}")
    print(f"\nIndicators: {data['indicator'].unique()}")
    print(f"\nCountries: {data['country'].unique()}")
    print(f"\nDate range: {data['date'].min()} to {data['date'].max()}")
