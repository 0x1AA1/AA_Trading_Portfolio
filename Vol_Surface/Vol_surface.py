import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "Algo_Trade_IBKR" / "ibkr_api"))

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try importing yfinance for public data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not installed. Install with: pip install yfinance")

# Try importing IBKR connector
try:
    from ibkr_connector import OptionsDataFetcher
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    print("Warning: IBKR connector not available")

class VolatilitySurface:
    def __init__(self, ticker, data_source='yfinance', sec_type='STK', exchange='SMART', currency='USD'):
        """
        Initialize the volatility surface extractor for a given ticker

        Args:
            ticker: Symbol (e.g., 'AAPL', 'CL')
            data_source: 'yfinance' for public data or 'ibkr' for Interactive Brokers
            sec_type: Security type ('STK' for stocks, 'FUT' for futures) - used with IBKR
            exchange: Exchange (default 'SMART', 'NYMEX' for CL) - used with IBKR
            currency: Currency (default 'USD') - used with IBKR
        """
        self.ticker = ticker.upper()
        self.data_source = data_source.lower()
        self.sec_type = sec_type
        self.exchange = exchange
        self.currency = currency
        self.stock_price = None
        self.risk_free_rate = 0.05  # Default 5% - you can update this
        self.volatility_surface = None
        self.ibkr_fetcher = None

        # Initialize data source specific objects
        if self.data_source == 'yfinance':
            if not YFINANCE_AVAILABLE:
                raise ImportError("yfinance not installed. Install with: pip install yfinance")
            self.stock = yf.Ticker(self.ticker)
        elif self.data_source == 'ibkr':
            if not IBKR_AVAILABLE:
                raise ImportError("IBKR connector not available. Check ibkr_connector.py")

    def get_stock_price(self):
        """Get current stock price from selected data source"""
        if self.data_source == 'yfinance':
            try:
                info = self.stock.info
                self.stock_price = info.get('currentPrice') or info.get('regularMarketPrice')
                if not self.stock_price:
                    # Fallback to recent price data
                    hist = self.stock.history(period="1d")
                    self.stock_price = hist['Close'].iloc[-1]
                return self.stock_price
            except Exception as e:
                print(f"Error getting stock price from yfinance: {e}")
                return None
        elif self.data_source == 'ibkr':
            if self.volatility_surface is not None and not self.volatility_surface.empty:
                return self.stock_price
            return None
    
    def black_scholes_call(self, S, K, T, r, sigma):
        """
        Calculate Black-Scholes call option price
        """
        if T <= 0:
            return max(S - K, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    def black_scholes_put(self, S, K, T, r, sigma):
        """
        Calculate Black-Scholes put option price
        """
        if T <= 0:
            return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    
    def implied_volatility(self, option_price, S, K, T, r, option_type='call'):
        """
        Calculate implied volatility using Brent's method
        """
        if T <= 0:
            return np.nan
        
        def objective_function(sigma):
            if option_type.lower() == 'call':
                return self.black_scholes_call(S, K, T, r, sigma) - option_price
            else:
                return self.black_scholes_put(S, K, T, r, sigma) - option_price
        
        try:
            iv = brentq(objective_function, 0.001, 5.0, maxiter=100)
            return iv
        except:
            return np.nan
    
    def get_options_data_yfinance(self):
        """
        Fetch options data from Yahoo Finance (public data)
        """
        try:
            expirations = self.stock.options
            if not expirations:
                print(f"No options data available for {self.ticker}")
                return None

            all_options = []

            for exp_date in expirations[:8]:  # Limit to first 8 expirations for performance
                try:
                    opt_chain = self.stock.option_chain(exp_date)

                    # Process calls
                    calls = opt_chain.calls.copy()
                    calls['option_type'] = 'call'
                    calls['expiration'] = exp_date

                    # Process puts
                    puts = opt_chain.puts.copy()
                    puts['option_type'] = 'put'
                    puts['expiration'] = exp_date

                    # Combine calls and puts
                    options = pd.concat([calls, puts], ignore_index=True)
                    all_options.append(options)

                except Exception as e:
                    print(f"Error fetching options for {exp_date}: {e}")
                    continue

            if not all_options:
                print("No valid options data found")
                return None

            return pd.concat(all_options, ignore_index=True)

        except Exception as e:
            print(f"Error fetching options data from yfinance: {e}")
            return None

    def get_options_data_ibkr(self, use_cache=True, cache_hours=1):
        """
        Fetch options data from IBKR API

        Args:
            use_cache: Whether to use cached data if available
            cache_hours: Maximum cache age in hours

        Returns:
            DataFrame with options data
        """
        try:
            print(f"Fetching options data from IBKR for {self.ticker}...")

            # Create IBKR fetcher
            self.ibkr_fetcher = OptionsDataFetcher()

            # Connect to IBKR
            if not self.ibkr_fetcher.connect():
                print("Failed to connect to IBKR. Ensure TWS/Gateway is running.")
                return None

            # Check cache first
            if use_cache:
                cached = self.ibkr_fetcher.get_cached_options(
                    self.ticker,
                    max_age_hours=cache_hours
                )
                if cached is not None:
                    print(f"Using cached data for {self.ticker}")
                    # Transform to expected format
                    return self._transform_ibkr_data(cached)

            # Fetch fresh data from IBKR
            df = self.ibkr_fetcher.get_option_chain(
                symbol=self.ticker,
                sec_type=self.sec_type,
                exchange=self.exchange,
                currency=self.currency,
                strike_range=0.15,
                max_expiries=8
            )

            if df.empty:
                print(f"No options data available for {self.ticker}")
                self.ibkr_fetcher.disconnect()
                return None

            # Save to cache
            self.ibkr_fetcher.save_option_chain(df, self.ticker)

            # Disconnect
            self.ibkr_fetcher.disconnect()

            # Transform to expected format
            return self._transform_ibkr_data(df)

        except Exception as e:
            print(f"Error fetching options data from IBKR: {e}")
            if self.ibkr_fetcher:
                self.ibkr_fetcher.disconnect()
            return None

    def get_options_data(self, use_cache=True, cache_hours=1):
        """
        Fetch options data from selected data source

        Args:
            use_cache: Whether to use cached data (IBKR only)
            cache_hours: Maximum cache age in hours (IBKR only)

        Returns:
            DataFrame with options data
        """
        if self.data_source == 'yfinance':
            return self.get_options_data_yfinance()
        elif self.data_source == 'ibkr':
            return self.get_options_data_ibkr(use_cache=use_cache, cache_hours=cache_hours)
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")

    def _transform_ibkr_data(self, df):
        """
        Transform IBKR data format to match expected format

        IBKR format: symbol, expiry, strike, right, bid, ask, iv, delta, gamma, etc.
        Expected format: strike, expiration, option_type, bid, ask, volume, etc.
        """
        transformed = df.copy()

        # Rename columns to match expected format
        column_mapping = {
            'right': 'option_type',
            'expiry': 'expiration',
            'last': 'lastPrice'
        }
        transformed.rename(columns=column_mapping, inplace=True)

        # Convert option type to lowercase
        transformed['option_type'] = transformed['option_type'].str.lower()

        # Ensure required columns exist
        if 'volume' not in transformed.columns:
            transformed['volume'] = transformed.get('volume', 100)  # Default volume

        return transformed
    
    def calculate_time_to_expiry(self, expiration_date):
        """
        Calculate time to expiry in years
        """
        exp_date = pd.to_datetime(expiration_date)
        current_date = pd.to_datetime(datetime.now().date())
        days_to_expiry = (exp_date - current_date).days
        return max(days_to_expiry / 365.0, 1/365)  # Minimum 1 day
    
    def build_volatility_surface(self, use_cache=True, cache_hours=1):
        """
        Build the complete volatility surface using IBKR data

        Args:
            use_cache: Whether to use cached data if available
            cache_hours: Maximum cache age in hours

        Returns:
            DataFrame with complete volatility surface data
        """
        # Fetch options data from IBKR
        options_data = self.get_options_data(use_cache=use_cache, cache_hours=cache_hours)
        if options_data is None:
            return None

        # Extract stock price from IBKR data (use moneyness if available)
        if 'moneyness' in options_data.columns and not options_data.empty:
            # Calculate stock price from strike and moneyness
            sample_row = options_data.iloc[0]
            self.stock_price = sample_row['strike'] / sample_row['moneyness']
        else:
            # Calculate from ATM options
            atm_options = options_data[abs(options_data['strike'] - options_data['strike'].median()) < 5]
            if not atm_options.empty:
                self.stock_price = atm_options['strike'].median()
            else:
                self.stock_price = options_data['strike'].median()

        print(f"Current stock price for {self.ticker}: ${self.stock_price:.2f}")

        # Calculate time to expiry if not already present
        if 'time_to_expiry' not in options_data.columns:
            options_data['time_to_expiry'] = options_data['expiration'].apply(
                self.calculate_time_to_expiry
            )

        # Use IBKR implied volatility if available, otherwise calculate
        if 'iv' in options_data.columns:
            print("Using IBKR-provided implied volatilities...")
            options_data['implied_volatility'] = options_data['iv']

            # Filter out invalid IVs
            options_data = options_data[
                (options_data['implied_volatility'].notna()) &
                (options_data['implied_volatility'] > 0) &
                (options_data['implied_volatility'] < 5)
            ].copy()
        else:
            print("Calculating implied volatilities...")
            # Filter options with reasonable parameters first
            options_data = options_data[
                (options_data['time_to_expiry'] > 0) &
                (options_data['bid'] > 0) &
                (options_data['ask'] > 0)
            ].copy()

            if options_data.empty:
                print("No valid options data after filtering")
                return None

            # Calculate mid price if not present
            if 'mid' not in options_data.columns:
                options_data['mid'] = (options_data['bid'] + options_data['ask']) / 2

            # Calculate implied volatility
            implied_vols = []
            for idx, row in options_data.iterrows():
                iv = self.implied_volatility(
                    row['mid'],
                    self.stock_price,
                    row['strike'],
                    row['time_to_expiry'],
                    self.risk_free_rate,
                    row['option_type']
                )
                implied_vols.append(iv)

            options_data['implied_volatility'] = implied_vols

            # Remove rows where IV calculation failed
            options_data = options_data.dropna(subset=['implied_volatility'])

            if options_data.empty:
                print("No valid implied volatilities calculated")
                return None

        # Calculate moneyness if not present
        if 'moneyness' not in options_data.columns:
            options_data['moneyness'] = options_data['strike'] / self.stock_price

        # Store the surface
        self.volatility_surface = options_data

        print(f"Successfully built volatility surface with {len(options_data)} data points")
        return options_data
    
    def get_surface_summary(self):
        """
        Get summary statistics of the volatility surface
        """
        if self.volatility_surface is None:
            print("No volatility surface data available")
            return None
        
        summary = {
            'ticker': self.ticker,
            'stock_price': self.stock_price,
            'total_options': len(self.volatility_surface),
            'expiration_dates': sorted(self.volatility_surface['expiration'].unique()),
            'time_to_expiry_range': (
                self.volatility_surface['time_to_expiry'].min(),
                self.volatility_surface['time_to_expiry'].max()
            ),
            'strike_range': (
                self.volatility_surface['strike'].min(),
                self.volatility_surface['strike'].max()
            ),
            'iv_range': (
                self.volatility_surface['implied_volatility'].min(),
                self.volatility_surface['implied_volatility'].max()
            ),
            'avg_iv_by_type': self.volatility_surface.groupby('option_type')['implied_volatility'].mean().to_dict()
        }
        
        return summary
    
    def create_surface_pivot(self):
        """
        Create a pivot table for the volatility surface
        """
        if self.volatility_surface is None:
            return None
        
        # Create pivot table with strikes as rows and expiration dates as columns
        pivot_data = self.volatility_surface.pivot_table(
            index='strike',
            columns='expiration',
            values='implied_volatility',
            aggfunc='mean'
        )
        
        return pivot_data
    
    def plot_volatility_surface(self, save_plot=False):
        """
        Create comprehensive volatility surface visualizations
        """
        if self.volatility_surface is None:
            print("No volatility surface data to plot")
            return
        
        # Create subplot layout
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 3D Surface Plot
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        
        # Prepare data for 3D plot
        pivot_data = self.create_surface_pivot()
        if pivot_data is not None and not pivot_data.empty:
            X, Y = np.meshgrid(range(len(pivot_data.columns)), pivot_data.index)
            Z = pivot_data.values
            
            # Remove NaN values
            mask = ~np.isnan(Z)
            if mask.any():
                surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
                ax1.set_xlabel('Expiration Index')
                ax1.set_ylabel('Strike Price')
                ax1.set_zlabel('Implied Volatility')
                ax1.set_title(f'{self.ticker} - 3D Volatility Surface')
        
        # 2. Heatmap
        ax2 = fig.add_subplot(2, 3, 2)
        if pivot_data is not None and not pivot_data.empty:
            sns.heatmap(pivot_data, cmap='viridis', cbar=True, ax=ax2)
            ax2.set_title(f'{self.ticker} - IV Heatmap (Strike vs Expiration)')
            ax2.set_xlabel('Expiration Date')
            ax2.set_ylabel('Strike Price')
        
        # 3. IV vs Strike for different expirations
        ax3 = fig.add_subplot(2, 3, 3)
        expirations = sorted(self.volatility_surface['expiration'].unique())[:5]  # Top 5 expirations
        for exp in expirations:
            exp_data = self.volatility_surface[self.volatility_surface['expiration'] == exp]
            ax3.plot(exp_data['strike'], exp_data['implied_volatility'], 
                    'o-', label=exp, alpha=0.7)
        ax3.axvline(x=self.stock_price, color='red', linestyle='--', label='Current Price')
        ax3.set_xlabel('Strike Price')
        ax3.set_ylabel('Implied Volatility')
        ax3.set_title(f'{self.ticker} - IV vs Strike')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. IV vs Moneyness
        ax4 = fig.add_subplot(2, 3, 4)
        calls_data = self.volatility_surface[self.volatility_surface['option_type'] == 'call']
        puts_data = self.volatility_surface[self.volatility_surface['option_type'] == 'put']
        
        if not calls_data.empty:
            ax4.scatter(calls_data['moneyness'], calls_data['implied_volatility'], 
                       alpha=0.6, label='Calls', color='green')
        if not puts_data.empty:
            ax4.scatter(puts_data['moneyness'], puts_data['implied_volatility'], 
                       alpha=0.6, label='Puts', color='red')
        
        ax4.axvline(x=1.0, color='black', linestyle='--', label='ATM')
        ax4.set_xlabel('Moneyness (Strike/Spot)')
        ax4.set_ylabel('Implied Volatility')
        ax4.set_title(f'{self.ticker} - IV vs Moneyness')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Term Structure
        ax5 = fig.add_subplot(2, 3, 5)
        atm_data = self.volatility_surface[
            abs(self.volatility_surface['moneyness'] - 1.0) < 0.05
        ]
        if not atm_data.empty:
            term_structure = atm_data.groupby('time_to_expiry')['implied_volatility'].mean()
            ax5.plot(term_structure.index * 365, term_structure.values, 'o-', linewidth=2)
            ax5.set_xlabel('Days to Expiration')
            ax5.set_ylabel('Implied Volatility')
            ax5.set_title(f'{self.ticker} - ATM Term Structure')
            ax5.grid(True, alpha=0.3)
        
        # 6. Volume vs IV scatter
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.scatter(self.volatility_surface['volume'], 
                   self.volatility_surface['implied_volatility'], 
                   alpha=0.6, c=self.volatility_surface['time_to_expiry'], 
                   cmap='plasma')
        ax6.set_xlabel('Volume')
        ax6.set_ylabel('Implied Volatility')
        ax6.set_title(f'{self.ticker} - Volume vs IV')
        ax6.set_xscale('log')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(f'{self.ticker}_volatility_surface.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved as {self.ticker}_volatility_surface.png")
        
        plt.show()

def analyze_volatility_surface(ticker, data_source='yfinance', sec_type='STK', exchange='SMART',
                              currency='USD', use_cache=True):
    """
    Main function to analyze volatility surface for a given ticker

    Args:
        ticker: Symbol (e.g., 'AAPL', 'CL')
        data_source: 'yfinance' for public data or 'ibkr' for Interactive Brokers
        sec_type: Security type ('STK' for stocks, 'FUT' for futures) - used with IBKR
        exchange: Exchange (default 'SMART', 'NYMEX' for CL) - used with IBKR
        currency: Currency (default 'USD') - used with IBKR
        use_cache: Whether to use cached data if available (IBKR only)

    Returns:
        VolatilitySurface object with complete analysis
    """
    print(f"\n{'='*60}")
    print(f"VOLATILITY SURFACE ANALYSIS FOR {ticker.upper()}")
    print(f"Data Source: {data_source.upper()}")
    if data_source.lower() == 'ibkr':
        print(f"Security Type: {sec_type} | Exchange: {exchange}")
    print(f"{'='*60}")

    # Create volatility surface object
    vol_surface = VolatilitySurface(
        ticker=ticker,
        data_source=data_source,
        sec_type=sec_type,
        exchange=exchange,
        currency=currency
    )

    # Build the surface
    surface_data = vol_surface.build_volatility_surface(use_cache=use_cache)

    if surface_data is None:
        print(f"Unable to build volatility surface for {ticker}")
        return None

    # Get summary
    summary = vol_surface.get_surface_summary()
    if summary:
        print(f"\nSUMMARY:")
        print(f"Stock Price: ${summary['stock_price']:.2f}")
        print(f"Total Options: {summary['total_options']}")
        print(f"Expiration Dates: {len(summary['expiration_dates'])}")
        print(f"Time to Expiry Range: {summary['time_to_expiry_range'][0]:.3f} - {summary['time_to_expiry_range'][1]:.3f} years")
        print(f"Strike Range: ${summary['strike_range'][0]:.2f} - ${summary['strike_range'][1]:.2f}")
        print(f"IV Range: {summary['iv_range'][0]:.1%} - {summary['iv_range'][1]:.1%}")
        print(f"Average IV - Calls: {summary['avg_iv_by_type'].get('call', 0):.1%}")
        print(f"Average IV - Puts: {summary['avg_iv_by_type'].get('put', 0):.1%}")

    # Create visualizations
    print(f"\nGenerating volatility surface plots...")
    vol_surface.plot_volatility_surface(save_plot=True)

    # Save to cache directory
    cache_dir = Path(__file__).parent.parent / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_file = cache_dir / f'{ticker}_volatility_surface_{data_source}.csv'
    vol_surface.volatility_surface.to_csv(output_file, index=False)
    print(f"\nData saved to {output_file}")

    # Return the surface data for further analysis
    return vol_surface

# Example usage
if __name__ == "__main__":
    print("="*80)
    print("VOLATILITY SURFACE EXTRACTOR")
    print("="*80)
    print("\nChoose your data source:")
    print("  1. Yahoo Finance (Public Data) - Free, no setup required")
    print("  2. Interactive Brokers (IBKR API) - Real-time, requires TWS/Gateway")

    data_source_choice = input("\nSelect data source (1 or 2): ").strip()

    if data_source_choice == "2":
        data_source = "ibkr"
        print("\nIBKR selected. Ensure TWS/Gateway is running.")
    else:
        data_source = "yfinance"
        print("\nYahoo Finance selected.")

    print("\n" + "-"*80)

    # Symbol selection based on data source
    if data_source == "ibkr":
        examples = {
            "1": ("AAPL", "STK", "SMART", "Stock - Apple"),
            "2": ("SPY", "STK", "SMART", "ETF - S&P 500"),
            "3": ("CL", "FUT", "NYMEX", "Futures - Crude Oil"),
            "4": ("CUSTOM", "", "", "Custom Symbol")
        }

        print("\nSelect a symbol:")
        for key, (symbol, sec_type, exchange, desc) in examples.items():
            print(f"  {key}. {desc} ({symbol})")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "4":
            ticker = input("Enter ticker symbol: ").strip().upper()
            sec_type = input("Security type (STK/FUT/IND): ").strip().upper() or "STK"
            exchange = input("Exchange (SMART/NYMEX/etc): ").strip().upper() or "SMART"
            currency = input("Currency (USD): ").strip().upper() or "USD"
        elif choice in examples:
            ticker, sec_type, exchange, desc = examples[choice]
            currency = "USD"
            print(f"\nSelected: {desc}")
        else:
            print("Invalid choice, using AAPL as default")
            ticker, sec_type, exchange = "AAPL", "STK", "SMART"
            currency = "USD"

        # Ask about caching for IBKR
        use_cache_input = input("\nUse cached data if available? (Y/n): ").strip().lower()
        use_cache = use_cache_input != 'n'

    else:  # yfinance
        examples = ["AAPL", "TSLA", "SPY", "QQQ"]
        print("\nCommon tickers:", ", ".join(examples))
        ticker = input("\nEnter ticker symbol (or press Enter for AAPL): ").strip().upper()
        if not ticker:
            ticker = "AAPL"

        # Set defaults for yfinance
        sec_type, exchange, currency = "STK", "SMART", "USD"
        use_cache = False

    # Analyze the volatility surface
    print(f"\nStarting analysis for {ticker} using {data_source.upper()}...")
    print("-"*80)

    vol_surface = analyze_volatility_surface(
        ticker=ticker,
        data_source=data_source,
        sec_type=sec_type,
        exchange=exchange,
        currency=currency,
        use_cache=use_cache
    )

    if vol_surface and vol_surface.volatility_surface is not None:
        print(f"\nFirst 10 rows of volatility surface data:")
        display_cols = ['strike', 'expiration', 'option_type', 'implied_volatility', 'moneyness']
        if 'volume' in vol_surface.volatility_surface.columns:
            display_cols.append('volume')
        print(vol_surface.volatility_surface[display_cols].head(10))

        print(f"\nAnalysis complete for {ticker}")
    else:
        print(f"\nFailed to analyze volatility surface for {ticker}")
        if data_source == "ibkr":
            print("Check that TWS/Gateway is running and configured correctly.")