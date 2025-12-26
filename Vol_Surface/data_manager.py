"""
Options Data Manager

Unified interface for fetching options data from multiple sources:
- IBKR (Interactive Brokers) - real-time, professional data
- yfinance - free, public data

Provides caching, historical volatility calculation, and data normalization
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging

# Add IBKR API to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Algo_Trade_IBKR" / "ibkr_api"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptionsDataManager:
    """
    Manages options data fetching from multiple sources with caching
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize data manager

        Args:
            cache_dir: Directory for caching data
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "data_cache" / "vol_surface"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Data Manager initialized (cache: {self.cache_dir})")

    def fetch_options_data(
        self,
        ticker: str,
        data_source: str = 'yfinance',
        sec_type: str = 'STK',
        exchange: str = 'SMART',
        currency: str = 'USD',
        use_cache: bool = True,
        cache_minutes: int = 5
    ) -> pd.DataFrame:
        """
        Fetch options data from specified source

        Args:
            ticker: Symbol to fetch
            data_source: 'yfinance' or 'ibkr'
            sec_type: Security type (for IBKR)
            exchange: Exchange (for IBKR)
            currency: Currency (for IBKR)
            use_cache: Whether to use cached data
            cache_minutes: Maximum cache age in minutes (default 5 for intraday freshness)

        Returns:
            DataFrame with options data (normalized format)
        """
        logger.info(f"Fetching options data for {ticker} from {data_source}...")

        # Check cache first
        if use_cache:
            cached_df = self._load_cache(ticker, data_source, cache_minutes)
            if cached_df is not None:
                return cached_df

        # Fetch fresh data
        if data_source.lower() == 'yfinance':
            df = self._fetch_yfinance(ticker)
        elif data_source.lower() == 'ibkr':
            df = self._fetch_ibkr(ticker, sec_type, exchange, currency)
        else:
            raise ValueError(f"Unsupported data source: {data_source}")

        # Normalize format
        df = self._normalize_dataframe(df, data_source)

        # Cache the data
        if not df.empty:
            self._save_cache(df, ticker, data_source)

        return df

    def _fetch_yfinance(self, ticker: str) -> pd.DataFrame:
        """Fetch from yfinance"""
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            return pd.DataFrame()

        try:
            stock = yf.Ticker(ticker)

            # Get all expiration dates
            expirations = stock.options

            if not expirations:
                logger.warning(f"No options available for {ticker}")
                return pd.DataFrame()

            all_options = []

            for exp_date in expirations:
                try:
                    opt_chain = stock.option_chain(exp_date)

                    # Calls
                    calls = opt_chain.calls.copy()
                    calls['option_type'] = 'call'
                    calls['expiration'] = exp_date

                    # Puts
                    puts = opt_chain.puts.copy()
                    puts['option_type'] = 'put'
                    puts['expiration'] = exp_date

                    all_options.append(calls)
                    all_options.append(puts)

                except Exception as e:
                    logger.warning(f"Failed to fetch {exp_date}: {e}")
                    continue

            if not all_options:
                return pd.DataFrame()

            df = pd.concat(all_options, ignore_index=True)

            logger.info(f"Fetched {len(df)} options from yfinance")
            return df

        except Exception as e:
            logger.error(f"Error fetching from yfinance: {e}")
            return pd.DataFrame()

    def _fetch_ibkr(
        self,
        ticker: str,
        sec_type: str = 'STK',
        exchange: str = 'SMART',
        currency: str = 'USD'
    ) -> pd.DataFrame:
        """Fetch from IBKR with safe connection handling"""
        try:
            from ibkr_connector import OptionsDataFetcher
            import time
        except ImportError:
            logger.error("IBKR connector not found. Check Algo_Trade_IBKR/ibkr_api")
            return pd.DataFrame()

        fetcher = None
        try:
            logger.info(f"Connecting to IBKR for {ticker} options data...")
            logger.info(f"Using clientId=10 to avoid conflicts with existing connections")

            # Create custom config with different clientId to avoid conflicts
            import json
            from pathlib import Path

            config_path = Path(__file__).parent.parent / "data_cache" / "ibkr_config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Use different clientId to avoid conflicts
            original_client_id = config['connection']['clientId']
            config['connection']['clientId'] = 10  # Different from default

            # Save temp config
            temp_config_path = Path(__file__).parent.parent / "data_cache" / "ibkr_config_temp_volsurface.json"
            with open(temp_config_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Create fetcher with custom config
            fetcher = OptionsDataFetcher(config_path=str(temp_config_path))

            # Connect with retries
            connected = fetcher.connect(retry_count=2, retry_delay=3)

            if not connected:
                logger.error("Could not connect to IBKR. Is TWS/Gateway running?")
                return pd.DataFrame()

            logger.info("Successfully connected to IBKR")

            # Fetch options data
            df = fetcher.get_option_chain(
                symbol=ticker,
                sec_type=sec_type,
                exchange=exchange,
                currency=currency,
                strike_range=0.15,  # +/- 15% from current price
                max_expiries=6
            )

            logger.info(f"Fetched {len(df)} options from IBKR")

            # Disconnect cleanly
            time.sleep(1)  # Give time for pending requests
            fetcher.disconnect()
            logger.info("Disconnected from IBKR")

            # Cleanup temp config
            if temp_config_path.exists():
                temp_config_path.unlink()

            return df

        except Exception as e:
            logger.error(f"Error fetching from IBKR: {e}")
            logger.error(f"Error type: {type(e).__name__}")

            # Ensure cleanup
            if fetcher and fetcher.is_connected():
                try:
                    fetcher.disconnect()
                    logger.info("Disconnected from IBKR after error")
                except:
                    pass

            # Cleanup temp config
            temp_config_path = Path(__file__).parent.parent / "data_cache" / "ibkr_config_temp_volsurface.json"
            if temp_config_path.exists():
                try:
                    temp_config_path.unlink()
                except:
                    pass

            return pd.DataFrame()

    def _normalize_dataframe(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Normalize dataframe to consistent format

        Standard columns:
        - strike: Strike price
        - expiration: Expiration date
        - option_type: 'call' or 'put'
        - bid: Bid price
        - ask: Ask price
        - last: Last traded price
        - volume: Trading volume
        - openInterest: Open interest
        - impliedVolatility: Implied volatility
        - delta, gamma, theta, vega (Greeks, if available)
        """
        if df.empty:
            return df

        df = df.copy()

        # Standardize column names based on source
        if source == 'ibkr':
            rename_map = {
                'right': 'option_type',
                'expiry': 'expiration',
                'iv': 'impliedVolatility',
                'bidSize': 'bidSize',
                'askSize': 'askSize'
            }
            df = df.rename(columns=rename_map)

            # Convert option_type format
            if 'option_type' in df.columns:
                df['option_type'] = df['option_type'].map({'C': 'call', 'P': 'put'})

        elif source == 'yfinance':
            # yfinance already uses good column names
            pass

        # Ensure required columns exist
        required_cols = ['strike', 'expiration', 'option_type', 'bid', 'ask']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing required column: {col}")

        # Calculate mid price if missing
        if 'mid' not in df.columns and 'bid' in df.columns and 'ask' in df.columns:
            df['mid'] = (df['bid'] + df['ask']) / 2

        # Convert expiration to datetime
        if 'expiration' in df.columns:
            df['expiration'] = pd.to_datetime(df['expiration'])

        # Calculate days to expiration
        if 'expiration' in df.columns:
            df['dte'] = (df['expiration'] - pd.Timestamp.now()).dt.days

        # Filter out expired or very near-term options (< 7 days)
        if 'dte' in df.columns:
            df = df[df['dte'] >= 7].copy()

        # Remove options with no bid/ask
        if 'bid' in df.columns and 'ask' in df.columns:
            df = df[(df['bid'] > 0) & (df['ask'] > 0)].copy()

        # Apply data quality filters
        df = self._apply_quality_filters(df)

        # Sort by expiration and strike
        if 'expiration' in df.columns and 'strike' in df.columns:
            df = df.sort_values(['expiration', 'strike']).reset_index(drop=True)

        logger.info(f"Normalized {len(df)} options (format: {source})")

        return df

    def _apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data quality filters to remove illiquid/unreliable options

        Filters:
        - Minimum volume and open interest (liquidity)
        - Maximum bid-ask spread (pricing quality)
        - Non-zero mid price
        - Valid implied volatility
        """
        if df.empty:
            return df

        original_count = len(df)
        df = df.copy()

        # Filter 1: Liquidity - volume OR open interest must be reasonable
        # Relaxed for commodities: commodities have lower absolute volume but high turnover
        if 'volume' in df.columns and 'openInterest' in df.columns:
            # Keep if volume > 1 OR openInterest > 10 (commodity-friendly criteria)
            df = df[
                ((df['volume'].fillna(0) > 1) | (df['openInterest'].fillna(0) > 10))
            ].copy()

        # Filter 2: Bid-ask spread - must be < 100% of mid price
        # Relaxed for commodities: wider spreads due to commodity volatility and market structure
        if 'bid' in df.columns and 'ask' in df.columns and 'mid' in df.columns:
            df['spread_pct'] = ((df['ask'] - df['bid']) / df['mid']).abs()
            df = df[df['spread_pct'] < 1.00].copy()  # 100% max spread
            df = df.drop(columns=['spread_pct'])

        # Filter 3: Non-zero mid price
        if 'mid' in df.columns:
            df = df[df['mid'] > 0.01].copy()  # At least 1 cent

        # Filter 4: Valid implied volatility (if available)
        if 'impliedVolatility' in df.columns:
            # Remove zero or extremely high IV (likely bad data)
            df = df[
                (df['impliedVolatility'] > 0.01) &  # > 1%
                (df['impliedVolatility'] < 5.0)      # < 500%
            ].copy()

        filtered_count = original_count - len(df)
        if filtered_count > 0:
            logger.info(f"Quality filters removed {filtered_count} illiquid/unreliable options")

        return df

    def get_current_price(self, ticker: str, data_source: str = 'yfinance') -> float:
        """
        Get current price of underlying

        Args:
            ticker: Symbol
            data_source: 'yfinance' or 'ibkr'

        Returns:
            Current price
        """
        if data_source == 'yfinance':
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                info = stock.info
                price = info.get('currentPrice') or info.get('regularMarketPrice')

                if price is None:
                    # Try fast_info
                    price = stock.fast_info.get('lastPrice')

                if price is None:
                    # Fallback to history
                    hist = stock.history(period='1d')
                    if not hist.empty:
                        price = hist['Close'].iloc[-1]

                return float(price) if price else 0.0

            except Exception as e:
                logger.error(f"Error getting price from yfinance: {e}")
                return 0.0

        elif data_source == 'ibkr':
            try:
                from ibkr_connector import IBKRConnector
                from ib_insync import Stock

                with IBKRConnector() as ib:
                    contract = Stock(ticker, 'SMART', 'USD')
                    ib.ib.qualifyContracts(contract)

                    ticker_data = ib.ib.reqMktData(contract, '', False, False)
                    ib.ib.sleep(2)

                    if ticker_data.last:
                        return float(ticker_data.last)
                    elif ticker_data.close:
                        return float(ticker_data.close)
                    else:
                        logger.warning("Could not get current price from IBKR")
                        return 0.0

            except Exception as e:
                logger.error(f"Error getting price from IBKR: {e}")
                return 0.0

        else:
            raise ValueError(f"Unsupported data source: {data_source}")

    def calculate_historical_volatility(
        self,
        ticker: str,
        window: int = 30,
        data_source: str = 'yfinance'
    ) -> float:
        """
        Calculate historical volatility (annualized)

        Args:
            ticker: Symbol
            window: Lookback window in days
            data_source: Data source

        Returns:
            Annualized historical volatility
        """
        if data_source == 'yfinance':
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)

                # Get historical data
                hist = stock.history(period=f'{window+10}d')  # Extra days for safety

                if hist.empty or len(hist) < window:
                    logger.warning(f"Insufficient historical data for {ticker}")
                    return 0.0

                # Calculate log returns
                returns = np.log(hist['Close'] / hist['Close'].shift(1))
                returns = returns.dropna()

                # Annualized volatility
                volatility = returns.std() * np.sqrt(252)

                return float(volatility)

            except Exception as e:
                logger.error(f"Error calculating HV: {e}")
                return 0.0

        elif data_source == 'ibkr':
            # IBKR historical data would require more complex implementation
            # For now, fallback to yfinance
            logger.info("Using yfinance for historical volatility calculation")
            return self.calculate_historical_volatility(ticker, window, 'yfinance')

        else:
            raise ValueError(f"Unsupported data source: {data_source}")

    def _load_cache(
        self,
        ticker: str,
        source: str,
        max_age_minutes: int
    ) -> Optional[pd.DataFrame]:
        """Load cached data if fresh enough"""

        # New structure: cache under per-ticker folder
        ticker_dir = self.cache_dir / ticker.upper()
        cache_file = ticker_dir / f"{ticker}_{source}_options.csv"

        # Backward compatibility: fallback to legacy root file
        legacy_file = self.cache_dir / f"{ticker}_{source}_options.csv"

        if not cache_file.exists() and not legacy_file.exists():
            return None

        # Check age
        file_path = cache_file if cache_file.exists() else legacy_file
        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
        age = datetime.now() - file_time

        if age > timedelta(minutes=max_age_minutes):
            logger.info(f"Cache too old ({age}), fetching fresh data")
            return None

        try:
            df = pd.read_csv(file_path)

            # Convert expiration back to datetime
            if 'expiration' in df.columns:
                df['expiration'] = pd.to_datetime(df['expiration'])

            logger.info(f"Loaded {len(df)} options from cache (age: {age})")
            return df

        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None

    def _save_cache(self, df: pd.DataFrame, ticker: str, source: str):
        """Save data to cache"""

        ticker_dir = self.cache_dir / ticker.upper()
        ticker_dir.mkdir(parents=True, exist_ok=True)
        cache_file = ticker_dir / f"{ticker}_{source}_options.csv"

        try:
            df.to_csv(cache_file, index=False)
            logger.info(f"Cached {len(df)} options to {cache_file}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
