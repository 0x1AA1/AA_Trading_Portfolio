"""
Enhanced Volatility Surface Analysis with GAN, Greeks, and Arbitrage Strategies

Simple usage: Just input a ticker and get complete analysis
Enhanced with 2024-2025 research: GAN-based surface generation, volatility arbitrage, interactive Plotly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "Algo_Trade_IBKR" / "ibkr_api"))

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, timedelta
import warnings
import os
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

from greeks_calculator import GreeksCalculator, StraddleStrangle, VolatilityMetrics
from vol_arbitrage import VolatilityArbitrageStrategies
from plotly_visualizer import VolatilitySurfaceVisualizer
from international_data import InternationalTickerManager
from payoff_scenario_analysis import PayoffAnalyzer, create_common_strategy, AdvancedProbabilityAnalyzer

# Try to import GAN components
try:
    import torch
    from vol_gan import VolGAN, VolatilitySurfaceDataset
    from torch.utils.data import DataLoader
    GAN_AVAILABLE = True
except ImportError:
    GAN_AVAILABLE = False
    print("Note: PyTorch not available. GAN features disabled.")


class EnhancedVolatilitySurface:
    def __init__(self, ticker, data_source='yfinance', sec_type='STK', exchange='SMART', currency='USD',
                 risk_free_rate=0.05, dividend_yield=0.0, use_gan=False, use_cuda=False):
        """
        Initialize enhanced volatility surface analyzer

        Args:
            ticker: Stock ticker symbol
            data_source: 'yfinance' for public data or 'ibkr' for Interactive Brokers
            sec_type: Security type ('STK' for stocks, 'FUT' for futures) - used with IBKR
            exchange: Exchange (default 'SMART', 'NYMEX' for CL) - used with IBKR
            currency: Currency (default 'USD') - used with IBKR
            risk_free_rate: Risk-free interest rate (default 5%)
            dividend_yield: Dividend yield (default 0%)
            use_gan: Enable GAN-based surface generation (default False)
            use_cuda: Use CUDA GPU acceleration for GAN (default False)
        """
        self.ticker = ticker.upper()
        self.data_source = data_source.lower()
        self.sec_type = sec_type
        self.exchange = exchange
        self.currency = currency
        self.stock_price = None
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.volatility_surface = None
        self.use_gan = use_gan and GAN_AVAILABLE
        self.use_cuda = use_cuda and torch.cuda.is_available() if GAN_AVAILABLE else False
        self.ibkr_fetcher = None

        # Initialize data source specific objects
        if self.data_source == 'yfinance':
            if not YFINANCE_AVAILABLE:
                raise ImportError("yfinance not installed. Install with: pip install yfinance")
            self.stock = yf.Ticker(self.ticker)
        elif self.data_source == 'ibkr':
            if not IBKR_AVAILABLE:
                raise ImportError("IBKR connector not available. Check ibkr_connector.py")

        # GAN components
        self.gan_model = None
        self.historical_surfaces = []

        # Initialize new components
        self.greeks_calc = GreeksCalculator(risk_free_rate, dividend_yield)
        self.straddle_analyzer = StraddleStrangle(self.greeks_calc)
        self.arbitrage_analyzer = VolatilityArbitrageStrategies(self.greeks_calc)
        self.visualizer = VolatilitySurfaceVisualizer(theme='plotly_dark')
        self.ticker_mgr = InternationalTickerManager()

        if self.use_gan:
            print(f"GAN enabled. Device: {'CUDA GPU' if self.use_cuda else 'CPU'}")

    def get_stock_price(self):
        """Get current stock price from selected data source"""
        if self.data_source == 'yfinance':
            try:
                info = self.stock.info
                self.stock_price = info.get('currentPrice') or info.get('regularMarketPrice')
                if not self.stock_price:
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

    def implied_volatility(self, option_price, S, K, T, r, option_type='call'):
        """Calculate implied volatility using Brent's method"""
        if T <= 0 or option_price <= 0:
            return np.nan

        def objective_function(sigma):
            try:
                theo_price = self.greeks_calc.black_scholes_price(S, K, T, sigma, option_type)
                return theo_price - option_price
            except:
                return 1e10

        try:
            iv = brentq(objective_function, 0.001, 5.0, maxiter=100)
            return iv
        except:
            return np.nan

    def get_options_data_yfinance(self):
        """Fetch options data from Yahoo Finance"""
        try:
            expirations = self.stock.options
            if not expirations:
                print(f"No options data available for {self.ticker}")
                return None

            all_options = []

            for exp_date in expirations[:8]:
                try:
                    opt_chain = self.stock.option_chain(exp_date)

                    calls = opt_chain.calls.copy()
                    calls['option_type'] = 'call'
                    calls['expiration'] = exp_date

                    puts = opt_chain.puts.copy()
                    puts['option_type'] = 'put'
                    puts['expiration'] = exp_date

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
        """Fetch options data from IBKR API"""
        try:
            print(f"Fetching options data from IBKR for {self.ticker}...")
            self.ibkr_fetcher = OptionsDataFetcher()

            if not self.ibkr_fetcher.connect():
                print("Failed to connect to IBKR. Ensure TWS/Gateway is running.")
                return None

            if use_cache:
                cached = self.ibkr_fetcher.get_cached_options(self.ticker, max_age_hours=cache_hours)
                if cached is not None:
                    print(f"Using cached data for {self.ticker}")
                    return self._transform_ibkr_data(cached)

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

            self.ibkr_fetcher.save_option_chain(df, self.ticker)
            self.ibkr_fetcher.disconnect()
            return self._transform_ibkr_data(df)

        except Exception as e:
            print(f"Error fetching options data from IBKR: {e}")
            if self.ibkr_fetcher:
                self.ibkr_fetcher.disconnect()
            return None

    def _transform_ibkr_data(self, df):
        """Transform IBKR data format to match expected format"""
        transformed = df.copy()
        column_mapping = {
            'right': 'option_type',
            'expiry': 'expiration',
            'last': 'lastPrice'
        }
        transformed.rename(columns=column_mapping, inplace=True)
        transformed['option_type'] = transformed['option_type'].str.lower()
        if 'volume' not in transformed.columns:
            transformed['volume'] = transformed.get('volume', 100)
        return transformed

    def get_options_data(self, use_cache=True, cache_hours=1):
        """Fetch options data from selected data source"""
        if self.data_source == 'yfinance':
            return self.get_options_data_yfinance()
        elif self.data_source == 'ibkr':
            return self.get_options_data_ibkr(use_cache=use_cache, cache_hours=cache_hours)
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")

    def calculate_time_to_expiry(self, expiration_date):
        """Calculate time to expiry in years"""
        exp_date = pd.to_datetime(expiration_date)
        current_date = pd.to_datetime(datetime.now().date())
        days_to_expiry = (exp_date - current_date).days
        return max(days_to_expiry / 365.0, 1/365)

    def build_volatility_surface(self):
        """Build complete volatility surface with implied volatilities"""
        if not self.get_stock_price():
            print("Unable to get stock price")
            return None

        print(f"Current stock price for {self.ticker}: ${self.stock_price:.2f}")

        options_data = self.get_options_data()
        if options_data is None:
            return None

        # Calculate time to expiry
        options_data['time_to_expiry'] = options_data['expiration'].apply(self.calculate_time_to_expiry)

        # Filter valid options
        options_data = options_data[
            (options_data['time_to_expiry'] > 0) &
            (options_data['bid'] > 0) &
            (options_data['ask'] > 0)
        ].copy()

        if options_data.empty:
            print("No valid options data after filtering")
            return None

        # Calculate mid price
        options_data['mid_price'] = (options_data['bid'] + options_data['ask']) / 2

        # Calculate implied volatility
        print("Calculating implied volatilities...")
        implied_vols = []

        for idx, row in options_data.iterrows():
            iv = self.implied_volatility(
                row['mid_price'],
                self.stock_price,
                row['strike'],
                row['time_to_expiry'],
                self.risk_free_rate,
                row['option_type']
            )
            implied_vols.append(iv)

        options_data['implied_volatility'] = implied_vols
        options_data = options_data.dropna(subset=['implied_volatility'])

        if options_data.empty:
            print("No valid implied volatilities calculated")
            return None

        # Calculate moneyness
        options_data['moneyness'] = options_data['strike'] / self.stock_price

        self.volatility_surface = options_data
        return options_data

    def analyze_straddles(self):
        """Analyze straddle opportunities across different strikes"""
        if self.volatility_surface is None:
            return None

        print("\n" + "="*80)
        print("STRADDLE ANALYSIS")
        print("="*80)

        # Get nearest expiration with good liquidity
        expirations = sorted(self.volatility_surface['expiration'].unique())
        if not expirations:
            return None

        exp = expirations[0]
        exp_data = self.volatility_surface[self.volatility_surface['expiration'] == exp]
        T = exp_data['time_to_expiry'].iloc[0]

        # Find ATM strike
        strikes = sorted(exp_data['strike'].unique())
        atm_strike = min(strikes, key=lambda x: abs(x - self.stock_price))

        # Get ATM IV
        atm_calls = exp_data[(exp_data['strike'] == atm_strike) & (exp_data['option_type'] == 'call')]
        atm_puts = exp_data[(exp_data['strike'] == atm_strike) & (exp_data['option_type'] == 'put')]

        if atm_calls.empty or atm_puts.empty:
            return None

        call_iv = atm_calls.iloc[0]['implied_volatility']
        put_iv = atm_puts.iloc[0]['implied_volatility']
        avg_iv = (call_iv + put_iv) / 2

        # Market prices
        market_call = atm_calls.iloc[0]['mid_price']
        market_put = atm_puts.iloc[0]['mid_price']
        market_straddle = market_call + market_put

        # Theoretical analysis
        theo_price, components = self.straddle_analyzer.straddle_price(
            self.stock_price, atm_strike, T, avg_iv
        )
        lower_be, upper_be = self.straddle_analyzer.straddle_breakeven(
            self.stock_price, atm_strike, T, avg_iv
        )

        required_move = abs(upper_be - self.stock_price)
        required_move_pct = (required_move / self.stock_price) * 100

        print(f"\nExpiration: {exp} ({int(T*365)} days)")
        print(f"ATM Strike: ${atm_strike:.2f}")
        print(f"Average IV: {avg_iv:.2%}")
        print(f"\nMarket Straddle: ${market_straddle:.2f}")
        print(f"Theoretical Straddle: ${theo_price:.2f}")
        print(f"Difference: ${market_straddle - theo_price:.2f}")
        print(f"\nBreakeven Range: ${lower_be:.2f} - ${upper_be:.2f}")
        print(f"Required Move: ${required_move:.2f} ({required_move_pct:.2f}%)")

        # Greeks for the straddle
        greeks = self.greeks_calc.all_greeks(self.stock_price, atm_strike, T, avg_iv, 'call')
        print(f"\nStraddle Greeks:")
        print(f"  Delta: ~0 (by design)")
        print(f"  Gamma: {greeks['gamma']:.6f}")
        print(f"  Vega: {greeks['vega']*2:.4f} (call + put)")
        print(f"  Theta: {greeks['theta']*2:.4f} per day (call + put)")

        return {
            'expiration': exp,
            'strike': atm_strike,
            'market_price': market_straddle,
            'theo_price': theo_price,
            'breakevens': (lower_be, upper_be),
            'required_move_pct': required_move_pct,
            'greeks': greeks
        }

    def analyze_arbitrage_opportunities(self):
        """Identify volatility arbitrage opportunities"""
        if self.volatility_surface is None:
            return None

        print("\n" + "="*80)
        print("VOLATILITY ARBITRAGE ANALYSIS")
        print("="*80)

        opportunities = []

        # Calendar spread arbitrage
        expirations = sorted(self.volatility_surface['expiration'].unique())
        if len(expirations) >= 2:
            exp1, exp2 = expirations[0], expirations[1]

            exp1_data = self.volatility_surface[self.volatility_surface['expiration'] == exp1]
            exp2_data = self.volatility_surface[self.volatility_surface['expiration'] == exp2]

            T1 = exp1_data['time_to_expiry'].iloc[0]
            T2 = exp2_data['time_to_expiry'].iloc[0]

            # Common strikes
            common_strikes = set(exp1_data['strike']).intersection(set(exp2_data['strike']))

            for strike in list(common_strikes)[:5]:
                iv1 = exp1_data[exp1_data['strike'] == strike]['implied_volatility'].mean()
                iv2 = exp2_data[exp2_data['strike'] == strike]['implied_volatility'].mean()

                result = self.arbitrage_analyzer.calendar_spread_arbitrage(
                    self.stock_price, strike, T1, T2, iv1, iv2
                )

                if result['arbitrage_detected']:
                    opportunities.append({
                        'type': 'CALENDAR_SPREAD',
                        'strike': strike,
                        'result': result
                    })

        # Historical vs Implied vol
        hist_data = self.stock.history(period='30d')
        if len(hist_data) > 2:
            returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1)).dropna()
            hist_vol = returns.std() * np.sqrt(252)

            atm_data = self.volatility_surface[
                abs(self.volatility_surface['moneyness'] - 1.0) < 0.05
            ]

            if not atm_data.empty:
                implied_vol = atm_data['implied_volatility'].mean()
                exp = atm_data['expiration'].iloc[0]
                T = atm_data['time_to_expiry'].iloc[0]
                strike = atm_data['strike'].iloc[0]

                signal = self.arbitrage_analyzer.volatility_spread_strategy(
                    self.stock_price, strike, T, implied_vol, hist_vol, threshold=0.03
                )

                print(f"\nHistorical Vol (30d): {hist_vol:.2%}")
                print(f"Implied Vol (ATM): {implied_vol:.2%}")
                print(f"Vol Spread: {(implied_vol - hist_vol):.2%}")
                print(f"\nTrading Signal: {signal['signal']}")
                print(f"Action: {signal['action']}")
                print(f"Confidence: {signal['confidence']}")
                print(f"Expected P&L: ${signal['expected_pnl']:.2f}")

                if signal['signal'] != 'NO_TRADE':
                    opportunities.append({
                        'type': 'VOL_SPREAD',
                        'signal': signal
                    })

        if not opportunities:
            print("\nNo clear arbitrage opportunities detected")
        else:
            print(f"\n{len(opportunities)} potential opportunities identified")

        return opportunities

    def create_interactive_visualizations(self, output_folder=None):
        """Generate interactive Plotly visualizations"""
        if self.volatility_surface is None:
            return

        print("\n" + "="*80)
        print("GENERATING INTERACTIVE VISUALIZATIONS")
        print("="*80)

        # Use current directory if no output folder specified
        if output_folder is None:
            output_folder = os.getcwd()

        # Prepare data for 3D surface
        expirations = sorted(self.volatility_surface['expiration'].unique())[:6]
        strikes_range = self.volatility_surface['strike'].unique()
        strikes = sorted([s for s in strikes_range if 0.8*self.stock_price <= s <= 1.2*self.stock_price])

        if len(strikes) < 5 or len(expirations) < 2:
            print("Insufficient data for 3D visualization")
            return

        # Build IV surface grid
        maturities = []
        iv_surface = []

        for exp in expirations:
            exp_data = self.volatility_surface[self.volatility_surface['expiration'] == exp]
            T = exp_data['time_to_expiry'].iloc[0]
            maturities.append(T)

            iv_row = []
            for strike in strikes:
                strike_data = exp_data[exp_data['strike'] == strike]
                if not strike_data.empty:
                    iv_row.append(strike_data['implied_volatility'].mean())
                else:
                    iv_row.append(np.nan)
            iv_surface.append(iv_row)

        iv_surface = np.array(iv_surface).T
        maturities = np.array(maturities)
        strikes = np.array(strikes)

        # Remove any NaN rows
        valid_rows = ~np.isnan(iv_surface).all(axis=1)
        iv_surface = iv_surface[valid_rows]
        strikes = strikes[valid_rows]

        # 3D Surface
        print("\n1. Creating 3D volatility surface...")
        surface_path = os.path.join(output_folder, f"{self.ticker}_vol_surface_3d.html")
        fig_3d = self.visualizer.plot_3d_surface(
            strikes, maturities, iv_surface,
            spot_price=self.stock_price,
            title=f"{self.ticker} Implied Volatility Surface",
            save_html=surface_path
        )
        print(f"   Saved: {surface_path}")

        # Volatility smile (nearest expiration)
        print("\n2. Creating volatility smile...")
        exp = expirations[0]
        smile_data = self.volatility_surface[self.volatility_surface['expiration'] == exp]
        smile_strikes = sorted(smile_data['strike'].unique())
        smile_ivs = [smile_data[smile_data['strike'] == k]['implied_volatility'].mean()
                     for k in smile_strikes]

        smile_path = os.path.join(output_folder, f"{self.ticker}_vol_smile.html")
        fig_smile = self.visualizer.plot_smile_comparison(
            smile_strikes, [smile_ivs], ['Market IV'],
            spot_price=self.stock_price,
            title=f"{self.ticker} Volatility Smile - {exp}"
        )
        fig_smile.write_html(smile_path)
        print(f"   Saved: {smile_path}")

        # Term structure
        print("\n3. Creating ATM term structure...")
        atm_ivs = []
        term_maturities = []

        for exp in expirations:
            exp_data = self.volatility_surface[self.volatility_surface['expiration'] == exp]
            atm_data = exp_data[abs(exp_data['moneyness'] - 1.0) < 0.05]
            if not atm_data.empty:
                atm_ivs.append(atm_data['implied_volatility'].mean())
                term_maturities.append(atm_data['time_to_expiry'].iloc[0])

        if atm_ivs:
            term_path = os.path.join(output_folder, f"{self.ticker}_term_structure.html")
            fig_term = self.visualizer.plot_term_structure(
                np.array(term_maturities),
                np.array(atm_ivs),
                title=f"{self.ticker} ATM Volatility Term Structure"
            )
            fig_term.write_html(term_path)
            print(f"   Saved: {term_path}")

        print("\nAll visualizations generated successfully!")

    def train_gan_model(self, n_epochs=50, batch_size=16, output_folder=None):
        """
        Train GAN on current volatility surface

        Args:
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            output_folder: Folder to save the trained model
        """
        if output_folder is None:
            output_folder = os.getcwd()
        if not self.use_gan or self.volatility_surface is None:
            print("GAN training not available")
            return None

        print("\n" + "="*80)
        print("GAN TRAINING - Arbitrage-Free Surface Generation")
        print("="*80)

        # Prepare surface grid from current data
        expirations = sorted(self.volatility_surface['expiration'].unique())[:6]
        strikes_range = self.volatility_surface['strike'].unique()
        strikes = sorted([s for s in strikes_range if 0.8*self.stock_price <= s <= 1.2*self.stock_price])

        if len(strikes) < 10 or len(expirations) < 3:
            print("Insufficient data for GAN training (need at least 10 strikes and 3 expirations)")
            return None

        n_strikes = len(strikes)
        n_maturities = len(expirations)

        # Build IV surface
        maturities = []
        iv_surface_data = []

        for exp in expirations:
            exp_data = self.volatility_surface[self.volatility_surface['expiration'] == exp]
            T = exp_data['time_to_expiry'].iloc[0]
            maturities.append(T)

            iv_row = []
            for strike in strikes:
                strike_data = exp_data[exp_data['strike'] == strike]
                if not strike_data.empty:
                    iv_row.append(strike_data['implied_volatility'].mean())
                else:
                    iv_row.append(np.nan)
            iv_surface_data.append(iv_row)

        iv_surface_array = np.array(iv_surface_data).T

        # Interpolate NaNs
        for i in range(iv_surface_array.shape[0]):
            row = iv_surface_array[i, :]
            if np.isnan(row).any():
                mask = ~np.isnan(row)
                if mask.sum() > 1:
                    iv_surface_array[i, :] = np.interp(
                        np.arange(len(row)),
                        np.arange(len(row))[mask],
                        row[mask]
                    )

        # Store for GAN training
        self.historical_surfaces = [iv_surface_array]

        print(f"\nSurface shape: {n_strikes} strikes x {n_maturities} maturities")
        print(f"IV range: {iv_surface_array.min():.2%} - {iv_surface_array.max():.2%}")

        # Initialize GAN
        self.gan_model = VolGAN(
            latent_dim=100,
            n_strikes=n_strikes,
            n_maturities=n_maturities,
            lambda_calendar=10.0,
            lambda_butterfly=10.0
        )

        if self.use_cuda:
            self.gan_model.generator.cuda()
            self.gan_model.discriminator.cuda()

        # Prepare dataset
        surfaces_tensor = torch.FloatTensor([iv_surface_array])
        spot_prices_tensor = torch.FloatTensor([self.stock_price])
        strikes_tensor = torch.FloatTensor(strikes)
        maturities_tensor = torch.FloatTensor(maturities)

        dataset = VolatilitySurfaceDataset(
            surfaces_tensor, spot_prices_tensor, strikes_tensor, maturities_tensor
        )

        # Since we have only one surface, we'll use it multiple times with noise
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        print(f"\nTraining GAN for {n_epochs} epochs...")
        print(f"Device: {'CUDA' if self.use_cuda else 'CPU'}")

        # Train (simplified for single surface)
        history = self.gan_model.train(dataloader, n_epochs=n_epochs, verbose=True)

        print("\nGAN training complete!")
        print(f"Final Generator Loss: {history['gen_loss'][-1]:.4f}")
        print(f"Final Discriminator Loss: {history['disc_loss'][-1]:.4f}")

        # Save model
        model_path = os.path.join(output_folder, f"{self.ticker}_volgan_model.pth")
        self.gan_model.save(model_path)
        print(f"Model saved to: {model_path}")

        return history

    def generate_synthetic_surfaces(self, n_samples=5, output_folder=None):
        """
        Generate synthetic arbitrage-free volatility surfaces using trained GAN

        Args:
            n_samples: Number of surfaces to generate
            output_folder: Folder to save generated surface visualization

        Returns:
            Array of generated surfaces (n_samples, n_strikes, n_maturities)
        """
        if output_folder is None:
            output_folder = os.getcwd()
        if not self.use_gan or self.gan_model is None:
            print("GAN model not available. Train model first.")
            return None

        print("\n" + "="*80)
        print(f"GENERATING {n_samples} SYNTHETIC ARBITRAGE-FREE SURFACES")
        print("="*80)

        surfaces = self.gan_model.generate(self.stock_price, n_samples)

        print(f"\nGenerated {n_samples} surfaces")
        print(f"Shape: {surfaces.shape}")
        print(f"IV range: {surfaces.min():.2%} - {surfaces.max():.2%}")

        # Visualize one generated surface
        if len(self.historical_surfaces) > 0:
            sample_surface = surfaces[0]

            # Get strikes and maturities from last training
            expirations = sorted(self.volatility_surface['expiration'].unique())[:sample_surface.shape[1]]
            strikes_range = self.volatility_surface['strike'].unique()
            strikes = np.array(sorted([s for s in strikes_range if 0.8*self.stock_price <= s <= 1.2*self.stock_price])[:sample_surface.shape[0]])
            maturities = np.array([self.volatility_surface[self.volatility_surface['expiration'] == exp]['time_to_expiry'].iloc[0]
                                  for exp in expirations])

            print("\nVisualizing generated surface...")
            gan_surface_path = os.path.join(output_folder, f"{self.ticker}_gan_surface.html")
            fig = self.visualizer.plot_3d_surface(
                strikes, maturities, sample_surface,
                spot_price=self.stock_price,
                title=f"{self.ticker} - GAN Generated Arbitrage-Free Surface",
                save_html=gan_surface_path
            )
            print(f"Saved: {gan_surface_path}")

        return surfaces

    def launch_strategy_analysis(self, output_folder=None):
        """
        Launch interactive strategy analysis using actual market option premiums

        Args:
            output_folder: Folder to save analysis outputs
        """
        if output_folder is None:
            output_folder = os.getcwd()

        if self.volatility_surface is None:
            print("No volatility surface data available")
            return

        print("\n" + "="*80)
        print("OPTION STRATEGY PAYOFF & SCENARIO ANALYSIS")
        print("="*80)

        print("\nAnalysis modes:")
        print("  1. Build custom strategy (select individual options)")
        print("  2. Use common strategy template (straddle, spread, etc.)")
        print("  3. Analyze specific option from volatility surface")

        mode = input("\nSelect mode (1/2/3, default=3): ").strip()
        if not mode:
            mode = '3'

        if mode == '1':
            self._build_custom_strategy(output_folder)
        elif mode == '2':
            self._use_common_strategy(output_folder)
        elif mode == '3':
            self._analyze_specific_option(output_folder)
        else:
            print("Invalid mode selected")

    def _build_custom_strategy(self, output_folder):
        """Build custom multi-leg strategy"""
        print("\n" + "-"*80)
        print("CUSTOM STRATEGY BUILDER")
        print("-"*80)

        analyzer = PayoffAnalyzer(self.stock_price, self.risk_free_rate)

        while True:
            print("\nAdd option leg:")
            opt_type = input("  Option type (call/put, or 'done' to finish): ").strip().lower()
            if opt_type == 'done':
                break
            if opt_type not in ['call', 'put']:
                print("Invalid option type")
                continue

            try:
                strike = float(input("  Strike price: ").strip())
                premium = float(input("  Premium: ").strip())
                quantity = int(input("  Quantity (default=1): ").strip() or "1")
                position = input("  Position (long/short, default=long): ").strip().lower() or "long"

                analyzer.add_leg(opt_type, strike, premium, quantity, position)
                print(f"  Added {position} {quantity}x {opt_type} @ ${strike} for ${premium}")
            except ValueError:
                print("Invalid input")
                continue

        if len(analyzer.legs) == 0:
            print("No legs added")
            return

        self._run_payoff_analysis(analyzer, output_folder, "custom_strategy")

    def _use_common_strategy(self, output_folder):
        """Use pre-built strategy template"""
        print("\n" + "-"*80)
        print("COMMON STRATEGY TEMPLATES")
        print("-"*80)

        strategies = [
            "long_call", "long_put", "long_straddle", "short_straddle",
            "long_strangle", "short_strangle", "bull_call_spread", "bear_put_spread",
            "iron_condor", "butterfly"
        ]

        print("\nAvailable strategies:")
        for i, strategy in enumerate(strategies, 1):
            print(f"  {i}. {strategy.replace('_', ' ').title()}")

        try:
            choice = int(input("\nSelect strategy (1-10): ").strip())
            if choice < 1 or choice > len(strategies):
                print("Invalid choice")
                return

            strategy_name = strategies[choice - 1]

            # Get ATM premiums from volatility surface
            atm_data = self.volatility_surface[abs(self.volatility_surface['moneyness'] - 1.0) < 0.05]

            if atm_data.empty:
                print("No ATM options found in volatility surface")
                return

            atm_call_data = atm_data[atm_data['option_type'] == 'call']
            atm_put_data = atm_data[atm_data['option_type'] == 'put']

            atm_call_premium = atm_call_data.iloc[0]['mid_price'] if not atm_call_data.empty else None
            atm_put_premium = atm_put_data.iloc[0]['mid_price'] if not atm_put_data.empty else None

            offset = float(input("Strike offset from ATM (default=5): ").strip() or "5")

            analyzer = create_common_strategy(
                strategy_name,
                self.stock_price,
                atm_call_premium,
                atm_put_premium,
                offset
            )

            self._run_payoff_analysis(analyzer, output_folder, strategy_name)

        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
            return

    def _analyze_specific_option(self, output_folder):
        """Analyze specific option from volatility surface"""
        print("\n" + "-"*80)
        print("ANALYZE SPECIFIC OPTION")
        print("-"*80)

        # Show available expirations
        expirations = sorted(self.volatility_surface['expiration'].unique())
        print("\nAvailable expirations:")
        for i, exp in enumerate(expirations[:10], 1):
            days = int(self.volatility_surface[self.volatility_surface['expiration'] == exp]['time_to_expiry'].iloc[0] * 365)
            print(f"  {i}. {exp} ({days} days)")

        try:
            exp_choice = int(input("\nSelect expiration (1-{}): ".format(min(10, len(expirations)))).strip())
            if exp_choice < 1 or exp_choice > len(expirations):
                print("Invalid choice")
                return

            exp = expirations[exp_choice - 1]
            exp_data = self.volatility_surface[self.volatility_surface['expiration'] == exp]
            T = exp_data['time_to_expiry'].iloc[0]

            # Show available strikes
            strikes = sorted(exp_data['strike'].unique())
            print("\nAvailable strikes:")
            for i, strike in enumerate(strikes[:20], 1):
                print(f"  {i}. ${strike:.2f}")

            strike_choice = int(input("\nSelect strike (1-{}): ".format(min(20, len(strikes)))).strip())
            if strike_choice < 1 or strike_choice > len(strikes):
                print("Invalid choice")
                return

            strike = strikes[strike_choice - 1]

            opt_type = input("Option type (call/put): ").strip().lower()
            if opt_type not in ['call', 'put']:
                print("Invalid option type")
                return

            # Get option data
            option_data = exp_data[(exp_data['strike'] == strike) & (exp_data['option_type'] == opt_type)]

            if option_data.empty:
                print(f"No {opt_type} found at strike ${strike}")
                return

            option = option_data.iloc[0]
            premium = option['mid_price']
            iv = option['implied_volatility']

            print(f"\nSelected option:")
            print(f"  Type: {opt_type.upper()}")
            print(f"  Strike: ${strike:.2f}")
            print(f"  Premium: ${premium:.2f}")
            print(f"  Implied Vol: {iv:.2%}")
            print(f"  Expiration: {exp}")
            print(f"  Days to expiry: {int(T*365)}")

            position = input("\nPosition (long/short, default=long): ").strip().lower() or "long"
            quantity = int(input("Quantity (default=1): ").strip() or "1")

            # Advanced probability analysis
            print("\n" + "-"*80)
            print("Computing advanced probability metrics...")
            print("-"*80)

            # Get historical lookback period from user
            print("\nHistorical volatility calculation:")
            print("  Common periods: 30d, 60d, 90d, 6mo, 1y")
            lookback = input("Enter lookback period (default=30d): ").strip() or "30d"

            # Calculate historical volatility
            try:
                hist_data = self.stock.history(period=lookback)
                if len(hist_data) > 2:
                    returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1)).dropna()
                    historical_vol = returns.std() * np.sqrt(252)
                    historical_drift = returns.mean() * 252
                    print(f"  Using {len(hist_data)} days of historical data")
                else:
                    print(f"  Warning: Insufficient data for {lookback}, using IV as proxy")
                    historical_vol = iv
                    historical_drift = self.risk_free_rate
            except Exception as e:
                print(f"  Error fetching historical data: {e}")
                print(f"  Using IV as proxy for historical volatility")
                historical_vol = iv
                historical_drift = self.risk_free_rate

            # Create probability analyzer
            prob_analyzer = AdvancedProbabilityAnalyzer(self.stock_price, self.risk_free_rate)

            # Run comprehensive probability analysis
            prob_results = prob_analyzer.analyze_option_probabilities(
                strike, T, iv, historical_vol, historical_drift, opt_type
            )

            # Print probability report
            prob_analyzer.print_probability_report(prob_results)

            # Create analyzer
            analyzer = PayoffAnalyzer(self.stock_price, self.risk_free_rate)
            analyzer.add_leg(opt_type, strike, premium, quantity, position)

            # Run analysis
            self._run_payoff_analysis(analyzer, output_folder, f"{opt_type}_{strike}_{exp}", T, iv)

        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
            return

    def _run_payoff_analysis(self, analyzer, output_folder, strategy_name, time_to_expiry=None, volatility=None):
        """
        Run payoff diagram and Monte Carlo analysis

        Args:
            analyzer: PayoffAnalyzer instance
            output_folder: Output folder path
            strategy_name: Name for saved files
            time_to_expiry: Time to expiration (years), if None will prompt
            volatility: Implied volatility, if None will prompt
        """
        print("\n" + "-"*80)
        print("PAYOFF ANALYSIS")
        print("-"*80)

        # Payoff diagram
        print("\nGenerating payoff diagram...")
        payoff_path = os.path.join(output_folder, f"{strategy_name}_payoff.html")
        analyzer.plot_payoff(title=f"{strategy_name.replace('_', ' ').title()} - Payoff Diagram", save_html=payoff_path)
        print(f"Saved: {payoff_path}")

        # Calculate key metrics
        breakevens = analyzer.get_breakevens()
        max_profit = analyzer.get_max_profit()
        max_loss = analyzer.get_max_loss()

        print(f"\nKey Metrics:")
        if breakevens:
            print(f"  Breakeven points: {', '.join([f'${be:.2f}' for be in breakevens])}")
        print(f"  Max Profit: ${max_profit:.2f}" if max_profit != float('inf') else "  Max Profit: Unlimited")
        print(f"  Max Loss: ${max_loss:.2f}" if max_loss != -float('inf') else "  Max Loss: Unlimited")

        # Monte Carlo simulation
        run_mc = input("\nRun Monte Carlo simulation? (y/n, default=y): ").strip().lower()
        if run_mc != 'n':
            if time_to_expiry is None:
                time_to_expiry = float(input("Time to expiry (years, e.g., 0.25 for 3 months): ").strip())

            if volatility is None:
                # Use average IV from volatility surface
                atm_data = self.volatility_surface[abs(self.volatility_surface['moneyness'] - 1.0) < 0.05]
                volatility = atm_data['implied_volatility'].mean() if not atm_data.empty else 0.30
                print(f"Using volatility: {volatility:.2%}")

            n_sims = int(input("Number of simulations (default=10000): ").strip() or "10000")

            print(f"\nRunning {n_sims} Monte Carlo simulations...")
            mc_results = analyzer.monte_carlo_simulation(time_to_expiry, volatility, n_sims)

            print(f"\nMonte Carlo Results:")
            print(f"  Probability of Profit: {mc_results['prob_profit']:.2%}")
            print(f"  Expected P&L: ${mc_results['expected_value']:.2f}")
            print(f"  Value at Risk (95%): ${mc_results['var_95']:.2f}")
            print(f"  Conditional VaR (95%): ${mc_results['cvar_95']:.2f}")
            print(f"  Max Simulated Profit: ${mc_results['max_profit']:.2f}")
            print(f"  Max Simulated Loss: ${mc_results['max_loss']:.2f}")

            # Plot MC results
            print("\nGenerating Monte Carlo visualization...")
            mc_path = os.path.join(output_folder, f"{strategy_name}_monte_carlo.html")
            analyzer.plot_monte_carlo(
                mc_results,
                title=f"{strategy_name.replace('_', ' ').title()} - Monte Carlo Analysis",
                save_html=mc_path
            )
            print(f"Saved: {mc_path}")

        print("\nAnalysis complete!")

    def get_summary(self):
        """Get comprehensive summary"""
        if self.volatility_surface is None:
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


def create_output_folder(ticker):
    """
    Create organized folder structure: [TICKER]/[YYYY-QX-Date]/

    Args:
        ticker: Stock ticker symbol

    Returns:
        Path to output folder
    """
    now = datetime.now()

    # Determine quarter
    quarter = (now.month - 1) // 3 + 1

    # Format: 2025-Q3-20251007
    timestamp = f"{now.year}-Q{quarter}-{now.strftime('%Y%m%d')}"

    # Create folder structure
    base_path = os.path.join(os.getcwd(), ticker.upper())
    output_path = os.path.join(base_path, timestamp)

    os.makedirs(output_path, exist_ok=True)

    return output_path


def analyze_ticker(ticker, use_gan=False, use_cuda=False):
    """
    Main function - just input a ticker and get complete analysis

    Args:
        ticker: Stock ticker symbol
        use_gan: Enable GAN-based surface generation
        use_cuda: Use CUDA GPU acceleration for GAN
    """
    print("\n" + "="*80)
    print(f"ENHANCED VOLATILITY SURFACE ANALYSIS: {ticker.upper()}")
    print("="*80)

    # Create output folder
    output_folder = create_output_folder(ticker)
    print(f"\nOutput folder: {output_folder}")

    # Create analyzer
    vs = EnhancedVolatilitySurface(ticker, use_gan=use_gan, use_cuda=use_cuda)

    # Build surface
    surface_data = vs.build_volatility_surface()
    if surface_data is None:
        print(f"Unable to build volatility surface for {ticker}")
        return None

    # Summary
    summary = vs.get_summary()
    if summary:
        print(f"\nSUMMARY:")
        print(f"  Stock Price: ${summary['stock_price']:.2f}")
        print(f"  Total Options: {summary['total_options']}")
        print(f"  Expirations: {len(summary['expiration_dates'])}")
        print(f"  Time Range: {summary['time_to_expiry_range'][0]:.3f} - {summary['time_to_expiry_range'][1]:.3f} years")
        print(f"  Strike Range: ${summary['strike_range'][0]:.2f} - ${summary['strike_range'][1]:.2f}")
        print(f"  IV Range: {summary['iv_range'][0]:.1%} - {summary['iv_range'][1]:.1%}")
        print(f"  Avg IV (Calls): {summary['avg_iv_by_type'].get('call', 0):.1%}")
        print(f"  Avg IV (Puts): {summary['avg_iv_by_type'].get('put', 0):.1%}")

    # Straddle analysis
    vs.analyze_straddles()

    # Arbitrage opportunities
    vs.analyze_arbitrage_opportunities()

    # Interactive visualizations
    vs.create_interactive_visualizations(output_folder=output_folder)

    # Strategy payoff and scenario analysis
    analyze_strategy = input("\nAnalyze option strategies with payoff/scenario analysis? (y/n, default=n): ").strip().lower()
    if analyze_strategy == 'y':
        vs.launch_strategy_analysis(output_folder=output_folder)

    # GAN training and generation
    if use_gan:
        train_gan = input("\nTrain GAN model? (y/n, default=n): ").strip().lower()
        if train_gan == 'y':
            epochs = input("Number of training epochs (default=50): ").strip()
            epochs = int(epochs) if epochs else 50

            history = vs.train_gan_model(n_epochs=epochs, output_folder=output_folder)

            if history is not None:
                generate = input("\nGenerate synthetic surfaces? (y/n, default=y): ").strip().lower()
                if generate != 'n':
                    n_samples = input("Number of surfaces to generate (default=5): ").strip()
                    n_samples = int(n_samples) if n_samples else 5

                    synthetic_surfaces = vs.generate_synthetic_surfaces(n_samples, output_folder=output_folder)

                    if synthetic_surfaces is not None:
                        print(f"\n{n_samples} arbitrage-free surfaces generated successfully!")

    # Save data
    csv_path = os.path.join(output_folder, f'{ticker}_volatility_surface.csv')
    vs.volatility_surface.to_csv(csv_path, index=False)

    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_folder}")
    print(f"\nGenerated files:")
    print(f"  1. {ticker}_volatility_surface.csv - Complete options data")
    print(f"  2. {ticker}_vol_surface_3d.html - Interactive 3D surface")
    print(f"  3. {ticker}_vol_smile.html - Volatility smile chart")
    print(f"  4. {ticker}_term_structure.html - ATM term structure")

    if use_gan and vs.gan_model is not None:
        print(f"  5. {ticker}_volgan_model.pth - Trained GAN model")
        print(f"  6. {ticker}_gan_surface.html - GAN-generated surface")

    print(f"\nOpen the HTML files in your browser for interactive visualizations!")

    return vs


if __name__ == "__main__":
    print("="*80)
    print("ENHANCED VOLATILITY SURFACE ANALYZER")
    print("With GAN, Greeks, Arbitrage Analysis & Interactive Plotly Visualizations")
    print("="*80)

    # Example tickers
    examples = ["AAPL", "TSLA", "SPY", "NVDA", "MSFT", "GOOGL"]
    print(f"\nExample tickers: {', '.join(examples)}")

    # Get ticker from user
    ticker = input("\nEnter ticker symbol (or press Enter for AAPL): ").strip().upper()
    if not ticker:
        ticker = "AAPL"

    # Ask about GAN usage
    print("\n" + "-"*80)
    print("GAN CONFIGURATION (2024-2025 Research-Based)")
    print("-"*80)

    if not GAN_AVAILABLE:
        print("PyTorch not available. GAN features disabled.")
        print("To enable: pip install torch")
        use_gan = False
        use_cuda = False
    else:
        print("GAN enables arbitrage-free synthetic surface generation using")
        print("deep learning (VolGAN 2024-2025, arXiv:2304.13128)")
        print("\nFeatures:")
        print("  - Generate arbitrage-free synthetic surfaces")
        print("  - Calendar spread constraints (total variance monotonicity)")
        print("  - Butterfly spread constraints (convexity)")
        print("  - Stress testing scenarios")

        gan_choice = input("\nEnable GAN features? (y/n, default=n): ").strip().lower()
        use_gan = (gan_choice == 'y')

        if use_gan:
            import torch
            if torch.cuda.is_available():
                print(f"\nCUDA GPU detected: {torch.cuda.get_device_name(0)}")
                print("GPU acceleration can significantly speed up training")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

                cuda_choice = input("\nUse CUDA GPU acceleration? (y/n, default=y): ").strip().lower()
                use_cuda = (cuda_choice != 'n')
            else:
                print("\nCUDA GPU not available. Using CPU.")
                use_cuda = False
        else:
            use_cuda = False

    print("\n" + "-"*80)

    # Run analysis
    vs = analyze_ticker(ticker, use_gan=use_gan, use_cuda=use_cuda)

    if vs:
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nGenerated files:")
        print(f"  1. {ticker}_volatility_surface.csv - Complete data")
        print(f"  2. {ticker}_vol_surface_3d.html - Interactive 3D surface")
        print(f"  3. {ticker}_vol_smile.html - Volatility smile")
        print(f"  4. {ticker}_term_structure.html - ATM term structure")

        if use_gan and vs.gan_model is not None:
            print(f"  5. {ticker}_volgan_model.pth - Trained GAN model")
            print(f"  6. {ticker}_gan_surface.html - GAN-generated surface")

        print(f"\nOpen the HTML files in your browser for interactive visualizations!")
