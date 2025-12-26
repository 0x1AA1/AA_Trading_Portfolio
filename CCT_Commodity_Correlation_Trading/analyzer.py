# -*- coding: utf-8 -*-
"""
CCT Hourly Correlation Analyzer
Upgrades CCT to use hourly data for higher frequency trading signals
Includes ISIN/name enrichment for all outputs
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append('F:/1. perso - travail/2. Perso - Implementations/Coding/Python')
from utils.ticker_metadata import TickerMetadata

# Commodity universe
commodities = {
    'GC=F': 'Gold', 'SI=F': 'Silver', 'PL=F': 'Platinum', 'PA=F': 'Palladium',
    'HG=F': 'Copper', 'ALI=F': 'Aluminum', 'CL=F': 'WTI_Oil', 'BZ=F': 'Brent_Oil',
    'NG=F': 'NatGas', 'RB=F': 'Gasoline', 'ZW=F': 'Wheat', 'ZC=F': 'Corn'
}

# Equity universe
equities = {
    # Silver miners
    'CDE': 'Coeur', 'HL': 'Hecla', 'AG': 'FirstMajestic', 'EXK': 'Endeavour',
    'WPM': 'Wheaton', 'MAG': 'MAGSilver', 'PAAS': 'PanAmerican', 'FSM': 'Fortuna',
    # Gold miners
    'NEM': 'Newmont', 'GLD': 'SPDR_Gold_ETF', 'AEM': 'Agnico', 'KGC': 'Kinross',
    # Copper/diversified
    'FCX': 'Freeport', 'SCCO': 'SouthernCopper', 'BHP': 'BHP', 'RIO': 'RioTinto',
    # Oil & Gas
    'XOM': 'Exxon', 'CVX': 'Chevron', 'COP': 'ConocoPhillips', 'SLB': 'Schlumberger',
    # Oil refiners
    'VLO': 'Valero', 'MPC': 'Marathon', 'PSX': 'Phillips66', 'HF': 'HF_Sinclair'
}

print(f"Analyzing {len(commodities)} commodities x {len(equities)} equities = {len(commodities)*len(equities)} pairs")
print("Using HOURLY data for high-frequency correlation analysis")

# Download hourly data - yfinance max is 730 days for hourly, downloading maximum available
print("\nDownloading hourly data (last 730 days - maximum available for hourly interval)...")
all_tickers = list(commodities.keys()) + list(equities.keys())

try:
    # yfinance hourly data: use interval='1h' and period='730d' (API maximum for hourly)
    data = yf.download(
        all_tickers,
        period='730d',  # Last 730 days (yfinance maximum for hourly)
        interval='1h',   # Hourly data
        progress=False,
        auto_adjust=False,
        group_by='ticker'
    )
except Exception as e:
    print(f"Batch download error: {e}")
    # Fallback: download individually
    data = {}
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, period='730d', interval='1h', progress=False, auto_adjust=False)
            if not df.empty and 'Close' in df.columns:
                data[ticker] = df['Close']
        except Exception as err:
            print(f"  Failed to download {ticker}: {err}")

    data = pd.DataFrame(data)
    prices = data
else:
    # Extract Close prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data.xs('Close', axis=1, level=1)
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data

print(f"Downloaded {len(prices.columns)} tickers successfully")
print(f"Data points per ticker: {len(prices)} hours ({len(prices)/24:.1f} days)")

# Drop tickers with insufficient data (require at least 1000 hours = ~42 days)
min_required_hours = 1000
prices = prices.dropna(thresh=min_required_hours, axis=1)
print(f"Kept {len(prices.columns)} tickers with sufficient data (>={min_required_hours} hours)")

# Calculate log returns
returns = np.log(prices / prices.shift(1)).dropna()

# Calculate correlations
results = []
for comm_ticker, comm_name in commodities.items():
    if comm_ticker not in returns.columns:
        print(f"  Skipping {comm_name} - no data")
        continue

    for eq_ticker, eq_name in equities.items():
        if eq_ticker not in returns.columns:
            print(f"  Skipping {eq_name} - no data")
            continue

        comm_ret = returns[comm_ticker]
        eq_ret = returns[eq_ticker]

        # Contemporaneous correlation
        corr = comm_ret.corr(eq_ret)

        # Lag correlations (in hours)
        lag_corrs = {}
        for lag in [1, 3, 6, 12, 24]:  # 1h, 3h, 6h, 12h, 24h lags
            try:
                lag_corrs[f'lag{lag}h'] = comm_ret.corr(eq_ret.shift(-lag))
            except:
                lag_corrs[f'lag{lag}h'] = np.nan

        # Rolling correlation (24-hour window)
        try:
            rolling_corr = comm_ret.rolling(24).corr(eq_ret.rolling(24))
            rolling_mean = rolling_corr.mean()
            rolling_std = rolling_corr.std()
        except:
            rolling_mean = np.nan
            rolling_std = np.nan

        # Z-score analysis (60-hour rolling window)
        try:
            comm_z = (comm_ret - comm_ret.rolling(60).mean()) / comm_ret.rolling(60).std()
            z_eq_corr = comm_z.corr(eq_ret)
        except:
            z_eq_corr = np.nan

        results.append({
            'commodity_ticker': comm_ticker,
            'commodity': comm_name,
            'equity_ticker': eq_ticker,
            'equity': eq_name,
            'correlation': corr,
            'z_score_corr': z_eq_corr,
            'rolling_corr_mean': rolling_mean,
            'rolling_corr_std': rolling_std,
            **lag_corrs
        })

df = pd.DataFrame(results)
df = df.sort_values('correlation', ascending=False, key=lambda x: x.abs())

# Enrich with ISIN and names
print("\nEnriching with ISIN and full names...")
enricher = TickerMetadata()

# Enrich commodity side
comm_metadata = enricher.batch_get_metadata(df['commodity_ticker'].unique().tolist())
comm_metadata = comm_metadata.rename(columns={
    'ticker': 'commodity_ticker',
    'isin': 'commodity_isin',
    'name': 'commodity_name',
    'exchange': 'commodity_exchange',
    'currency': 'commodity_currency'
})

# Enrich equity side
eq_metadata = enricher.batch_get_metadata(df['equity_ticker'].unique().tolist())
eq_metadata = eq_metadata.rename(columns={
    'ticker': 'equity_ticker',
    'isin': 'equity_isin',
    'name': 'equity_name',
    'exchange': 'equity_exchange',
    'currency': 'equity_currency',
    'sector': 'equity_sector',
    'industry': 'equity_industry'
})

# Merge metadata
df = df.merge(comm_metadata[['commodity_ticker', 'commodity_isin', 'commodity_name', 'commodity_exchange']], on='commodity_ticker', how='left')
df = df.merge(eq_metadata[['equity_ticker', 'equity_isin', 'equity_name', 'equity_exchange', 'equity_sector', 'equity_industry']], on='equity_ticker', how='left')

# Reorder columns for readability
column_order = [
    'commodity_ticker', 'commodity', 'commodity_name', 'commodity_isin', 'commodity_exchange',
    'equity_ticker', 'equity', 'equity_name', 'equity_isin', 'equity_exchange', 'equity_sector', 'equity_industry',
    'correlation', 'z_score_corr', 'rolling_corr_mean', 'rolling_corr_std',
    'lag1h', 'lag3h', 'lag6h', 'lag12h', 'lag24h'
]

df = df[column_order]

# Save results
output_dir = Path('F:/1. perso - travail/2. Perso - Implementations/Coding/Python/CCT_Commodity_Correlation_Trading/results')
output_dir.mkdir(exist_ok=True, parents=True)
output_file = output_dir / 'hourly_correlations_enriched.csv'
df.to_csv(output_file, index=False)

# Generate reports
print(f"\n{'='*80}")
print("TOP 20 CORRELATIONS (Hourly Data)")
print(f"{'='*80}")
top20 = df.dropna(subset=['correlation']).head(20)
print(top20[['commodity', 'equity', 'equity_name', 'correlation', 'z_score_corr', 'lag1h', 'lag24h']].to_string(index=False))

print(f"\n{'='*80}")
print("OIL REFINER ANALYSIS (Hourly Sensitivity)")
print(f"{'='*80}")
oil_refiners = df[df['commodity'] == 'WTI_Oil']
oil_refiners = oil_refiners[oil_refiners['equity'].isin(['Valero', 'Marathon', 'Phillips66', 'HF_Sinclair'])]
print(oil_refiners[['equity', 'equity_name', 'correlation', 'lag1h', 'lag6h', 'lag24h']].to_string(index=False))

print(f"\n{'='*80}")
print("SILVER MINER TOP 5 (Hourly Correlations)")
print(f"{'='*80}")
silver_miners = df[df['commodity'] == 'Silver']
silver_top = silver_miners.dropna(subset=['correlation']).nlargest(5, 'correlation', keep='all')
print(silver_top[['equity', 'equity_name', 'equity_isin', 'correlation', 'z_score_corr', 'rolling_corr_std']].to_string(index=False))

print(f"\n{'='*80}")
print("HIGH-FREQUENCY TRADING OPPORTUNITIES")
print(f"{'='*80}")
# Identify pairs with strong contemporaneous correlation but weak lag correlation (mean reversion)
df['lag_differential'] = df['correlation'].abs() - df['lag24h'].abs()
mean_reversion_opps = df[
    (df['correlation'].abs() > 0.5) &
    (df['lag_differential'] > 0.1)
].sort_values('lag_differential', ascending=False).head(10)

print("\nMean Reversion Opportunities (Strong correlation, weak 24h lag):")
print(mean_reversion_opps[['commodity', 'equity', 'equity_name', 'correlation', 'lag24h', 'lag_differential']].to_string(index=False))

# Identify pairs with increasing correlation over time (momentum)
df['rolling_trend'] = df['rolling_corr_mean'] - df['correlation']
momentum_opps = df[
    (df['correlation'].abs() > 0.4) &
    (df['rolling_corr_std'] < 0.15)  # Low volatility in correlation
].sort_values('correlation', ascending=False).head(10)

print(f"\n\nMomentum Opportunities (Stable high correlation):")
print(momentum_opps[['commodity', 'equity', 'equity_name', 'correlation', 'rolling_corr_mean', 'rolling_corr_std']].to_string(index=False))

print(f"\n{'='*80}")
print(f"Results saved to: {output_file}")
print(f"Total pairs analyzed: {len(df)}")
print(f"Valid correlations: {df['correlation'].notna().sum()}")
print(f"Hourly data points: {len(prices)} hours ({len(prices)/24:.1f} days)")
print(f"{'='*80}")
