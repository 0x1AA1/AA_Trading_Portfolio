"""
Extract and clean trade data from Excel file.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the Excel file
excel_file = r"F:\1. perso - travail\2. Perso - Implementations\Coding\Python\Algo_Trade_1\Replicator\Full_Transac_vAA.XLSX"
df = pd.read_excel(excel_file, sheet_name='Full_transac')

print("=" * 80)
print("DETAILED TRADE DATA EXTRACTION")
print("=" * 80)

# Display full dataframe with all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("\nFull data (first 20 rows):")
print(df.head(20))

print("\n" + "=" * 80)
print("\nFull data (last 20 rows):")
print(df.tail(20))

print("\n" + "=" * 80)
print("\nUnique values in key columns:")

print(f"\nISIN unique values ({df['ISIN'].nunique()}):")
print(df['ISIN'].value_counts())

print(f"\nName unique values ({df['Name'].nunique()}):")
print(df['Name'].value_counts())

print(f"\nClass unique values ({df['Class'].nunique()}):")
print(df['Class'].value_counts())

print(f"\nStrategy unique values ({df['Strategy'].nunique()}):")
print(df['Strategy'].value_counts())

print(f"\nComment unique values ({df['Comment'].nunique()}):")
print(df['Comment'].value_counts())

# Filter to actual trades (non-null dates and valid quantities)
print("\n" + "=" * 80)
print("\nFiltering to actual trades...")
trades_df = df[df['Date'].notna() & df['Quantity'].notna() & (df['Quantity'] > 0)].copy()
print(f"Found {len(trades_df)} trades out of {len(df)} rows")

print("\nTrade summary:")
print(trades_df[['Date', 'Name', 'Quantity', 'Price_Cours', 'Class', 'Strategy', 'Comment']].to_string())

# Group by month
print("\n" + "=" * 80)
print("\nTrades grouped by month:")
trades_df['YearMonth'] = trades_df['Date'].dt.to_period('M')
monthly_trades = trades_df.groupby('YearMonth').agg({
    'Name': 'count',
    'Quantity': 'sum',
    'Value_At_Date (EUR)': lambda x: x.astype(str).str.replace(',', '').astype(float).sum() if x.notna().any() else 0
}).rename(columns={'Name': 'Trade_Count', 'Quantity': 'Total_Quantity', 'Value_At_Date (EUR)': 'Total_Value'})
print(monthly_trades)

# Asset distribution
print("\n" + "=" * 80)
print("\nAsset distribution:")
asset_dist = trades_df.groupby('Name').agg({
    'Quantity': 'sum',
    'Date': 'count'
}).rename(columns={'Quantity': 'Total_Quantity', 'Date': 'Trade_Count'})
print(asset_dist.sort_values('Trade_Count', ascending=False))

# Save cleaned trades
output_file = r"F:\1. perso - travail\2. Perso - Implementations\Coding\Python\Algo_Trade_1\Replicator\cleaned_trades.csv"
trades_df.to_csv(output_file, index=False)
print(f"\n\nCleaned trades saved to: {output_file}")

print("\n" + "=" * 80)
print("EXTRACTION COMPLETE")
print("=" * 80)
