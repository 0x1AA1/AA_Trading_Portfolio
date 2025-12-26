"""
International Exchange Support for Volatility Surface Analysis

Supports: Europe, Brazil, China, Israel, South Africa exchanges
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz


# Exchange configuration with ticker suffixes and metadata
EXCHANGE_CONFIG = {
    # European Exchanges
    'LSE': {
        'name': 'London Stock Exchange',
        'suffix': '.L',
        'timezone': 'Europe/London',
        'currency': 'GBP',
        'country': 'UK',
        'major_indices': ['FTSE', '^FTSE']
    },
    'EURONEXT_PARIS': {
        'name': 'Euronext Paris',
        'suffix': '.PA',
        'timezone': 'Europe/Paris',
        'currency': 'EUR',
        'country': 'France',
        'major_indices': ['CAC', '^FCHI']
    },
    'EURONEXT_AMSTERDAM': {
        'name': 'Euronext Amsterdam',
        'suffix': '.AS',
        'timezone': 'Europe/Amsterdam',
        'currency': 'EUR',
        'country': 'Netherlands',
        'major_indices': ['AEX', '^AEX']
    },
    'XETRA': {
        'name': 'Deutsche Börse XETRA',
        'suffix': '.DE',
        'timezone': 'Europe/Berlin',
        'currency': 'EUR',
        'country': 'Germany',
        'major_indices': ['DAX', '^GDAXI']
    },
    'SIX': {
        'name': 'SIX Swiss Exchange',
        'suffix': '.SW',
        'timezone': 'Europe/Zurich',
        'currency': 'CHF',
        'country': 'Switzerland',
        'major_indices': ['SMI', '^SSMI']
    },

    # South Africa
    'JSE': {
        'name': 'Johannesburg Stock Exchange',
        'suffix': '.JO',
        'timezone': 'Africa/Johannesburg',
        'currency': 'ZAR',
        'country': 'South Africa',
        'major_indices': ['JSE', 'J203.JO']
    },

    # Israel
    'TASE': {
        'name': 'Tel Aviv Stock Exchange',
        'suffix': '.TA',
        'timezone': 'Asia/Tel_Aviv',
        'currency': 'ILS',
        'country': 'Israel',
        'major_indices': ['TA-125', '^TA125.TA']
    },

    # Brazil
    'B3': {
        'name': 'B3 (Brasil Bolsa Balcão)',
        'suffix': '.SA',
        'timezone': 'America/Sao_Paulo',
        'currency': 'BRL',
        'country': 'Brazil',
        'major_indices': ['IBOVESPA', '^BVSP']
    },

    # China (Limited options data availability)
    'SSE': {
        'name': 'Shanghai Stock Exchange',
        'suffix': '.SS',
        'timezone': 'Asia/Shanghai',
        'currency': 'CNY',
        'country': 'China',
        'major_indices': ['SSE', '000001.SS']
    },
    'SZSE': {
        'name': 'Shenzhen Stock Exchange',
        'suffix': '.SZ',
        'timezone': 'Asia/Shanghai',
        'currency': 'CNY',
        'country': 'China',
        'major_indices': ['SZSE', '399001.SZ']
    },
    'HKEX': {
        'name': 'Hong Kong Stock Exchange',
        'suffix': '.HK',
        'timezone': 'Asia/Hong_Kong',
        'currency': 'HKD',
        'country': 'Hong Kong',
        'major_indices': ['HSI', '^HSI']
    },

    # US (for reference)
    'NYSE': {
        'name': 'New York Stock Exchange',
        'suffix': '',
        'timezone': 'America/New_York',
        'currency': 'USD',
        'country': 'USA',
        'major_indices': ['DJI', '^DJI']
    },
    'NASDAQ': {
        'name': 'NASDAQ',
        'suffix': '',
        'timezone': 'America/New_York',
        'currency': 'USD',
        'country': 'USA',
        'major_indices': ['IXIC', '^IXIC']
    }
}


# Popular stocks by exchange (with liquid options where available)
POPULAR_TICKERS = {
    'LSE': ['BP.L', 'HSBA.L', 'VOD.L', 'GSK.L', 'ULVR.L', 'AZN.L', 'RIO.L', 'SHEL.L'],
    'EURONEXT_PARIS': ['MC.PA', 'OR.PA', 'SAN.PA', 'AIR.PA', 'BNP.PA', 'TTE.PA'],
    'XETRA': ['SAP.DE', 'VOW3.DE', 'SIE.DE', 'BMW.DE', 'BAS.DE', 'ALV.DE'],
    'JSE': ['AGL.JO', 'SOL.JO', 'NPN.JO', 'SHP.JO', 'CFR.JO'],
    'TASE': ['TEVA.TA', 'ICL.TA', 'ESLT.TA'],
    'B3': ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA'],
    'HKEX': ['0700.HK', '0941.HK', '0005.HK', '0388.HK', '0939.HK'],
    'NASDAQ': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'],
    'NYSE': ['JPM', 'BAC', 'XOM', 'JNJ', 'PG', 'CVX']
}


class InternationalTickerManager:
    """
    Manage international ticker symbols and exchange information
    """

    def __init__(self):
        self.exchanges = EXCHANGE_CONFIG
        self.popular_tickers = POPULAR_TICKERS

    def format_ticker(self, base_ticker, exchange_code):
        """
        Format ticker with appropriate exchange suffix

        Args:
            base_ticker: Base ticker symbol (e.g., 'SAP')
            exchange_code: Exchange code (e.g., 'XETRA')

        Returns:
            Formatted ticker (e.g., 'SAP.DE')
        """
        if exchange_code not in self.exchanges:
            raise ValueError(f"Unknown exchange: {exchange_code}")

        suffix = self.exchanges[exchange_code]['suffix']

        # If ticker already has suffix, return as is
        if suffix and base_ticker.endswith(suffix):
            return base_ticker

        return f"{base_ticker}{suffix}"

    def get_exchange_info(self, exchange_code):
        """Get full exchange information"""
        return self.exchanges.get(exchange_code)

    def get_timezone(self, exchange_code):
        """Get timezone for exchange"""
        return self.exchanges[exchange_code]['timezone']

    def get_popular_tickers(self, exchange_code):
        """Get list of popular tickers for exchange"""
        return self.popular_tickers.get(exchange_code, [])

    def detect_exchange(self, ticker):
        """
        Detect exchange from ticker suffix

        Args:
            ticker: Full ticker symbol

        Returns:
            exchange_code or None
        """
        for code, config in self.exchanges.items():
            suffix = config['suffix']
            if suffix and ticker.endswith(suffix):
                return code

        # Default to US if no suffix
        return 'NASDAQ' if ticker.isupper() else None

    def get_market_hours(self, exchange_code):
        """
        Get typical market hours for exchange (in local time)

        Returns:
            dict with open and close times
        """
        market_hours = {
            'LSE': {'open': '08:00', 'close': '16:30'},
            'EURONEXT_PARIS': {'open': '09:00', 'close': '17:30'},
            'EURONEXT_AMSTERDAM': {'open': '09:00', 'close': '17:30'},
            'XETRA': {'open': '09:00', 'close': '17:30'},
            'SIX': {'open': '09:00', 'close': '17:30'},
            'JSE': {'open': '09:00', 'close': '17:00'},
            'TASE': {'open': '09:30', 'close': '17:25'},
            'B3': {'open': '10:00', 'close': '17:00'},
            'SSE': {'open': '09:30', 'close': '15:00'},
            'SZSE': {'open': '09:30', 'close': '15:00'},
            'HKEX': {'open': '09:30', 'close': '16:00'},
            'NYSE': {'open': '09:30', 'close': '16:00'},
            'NASDAQ': {'open': '09:30', 'close': '16:00'}
        }

        return market_hours.get(exchange_code, {'open': '09:00', 'close': '17:00'})

    def is_market_open(self, exchange_code):
        """
        Check if market is currently open

        Args:
            exchange_code: Exchange code

        Returns:
            bool: True if market is open
        """
        try:
            tz = pytz.timezone(self.get_timezone(exchange_code))
            now = datetime.now(tz)

            # Check if weekday (Monday=0, Sunday=6)
            if now.weekday() >= 5:  # Saturday or Sunday
                return False

            hours = self.get_market_hours(exchange_code)
            open_time = datetime.strptime(hours['open'], '%H:%M').time()
            close_time = datetime.strptime(hours['close'], '%H:%M').time()

            return open_time <= now.time() <= close_time

        except Exception:
            return False

    def get_currency_pair(self, from_currency, to_currency='USD'):
        """
        Get currency pair ticker for forex conversion

        Args:
            from_currency: Source currency (e.g., 'EUR')
            to_currency: Target currency (default 'USD')

        Returns:
            Currency pair ticker
        """
        if from_currency == to_currency:
            return None

        return f"{from_currency}{to_currency}=X"

    def convert_currency(self, amount, from_currency, to_currency='USD'):
        """
        Convert amount from one currency to another

        Args:
            amount: Amount to convert
            from_currency: Source currency
            to_currency: Target currency

        Returns:
            Converted amount
        """
        if from_currency == to_currency:
            return amount

        pair_ticker = self.get_currency_pair(from_currency, to_currency)

        try:
            pair = yf.Ticker(pair_ticker)
            rate = pair.history(period='1d')['Close'].iloc[-1]
            return amount * rate
        except Exception as e:
            print(f"Currency conversion failed: {e}")
            return amount

    def get_all_exchanges(self):
        """Get list of all supported exchanges"""
        return list(self.exchanges.keys())

    def get_exchanges_by_region(self):
        """Group exchanges by geographic region"""
        regions = {
            'Europe': ['LSE', 'EURONEXT_PARIS', 'EURONEXT_AMSTERDAM', 'XETRA', 'SIX'],
            'Asia': ['SSE', 'SZSE', 'HKEX', 'TASE'],
            'Africa': ['JSE'],
            'Americas': ['B3', 'NYSE', 'NASDAQ']
        }
        return regions

    def search_tickers(self, query, exchanges=None):
        """
        Search for tickers matching query across exchanges

        Args:
            query: Search string
            exchanges: List of exchange codes to search (None for all)

        Returns:
            List of matching tickers
        """
        if exchanges is None:
            exchanges = self.get_all_exchanges()

        matches = []
        query_upper = query.upper()

        for exchange in exchanges:
            tickers = self.get_popular_tickers(exchange)
            for ticker in tickers:
                if query_upper in ticker.upper():
                    matches.append({
                        'ticker': ticker,
                        'exchange': exchange,
                        'name': self.exchanges[exchange]['name'],
                        'currency': self.exchanges[exchange]['currency']
                    })

        return matches
