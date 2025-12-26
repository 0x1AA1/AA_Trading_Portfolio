"""
Analyze the historical trade data from Excel file to understand structure and content.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the Excel file
excel_file = r"F:\1. perso - travail\2. Perso - Implementations\Coding\Python\Algo_Trade_1\Replicator\Full_Transac_vAA.XLSX"

print("=" * 80)
print("TRADE DATA ANALYSIS")
print("=" * 80)

# Read Excel file - check all sheets
xl_file = pd.ExcelFile(excel_file)
print(f"\nSheets found: {xl_file.sheet_names}")

# Load each sheet and analyze
for sheet_name in xl_file.sheet_names:
    print(f"\n{'=' * 80}")
    print(f"SHEET: {sheet_name}")
    print(f"{'=' * 80}")

    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nBasic statistics:")
    print(df.describe())

    print("\nMissing values:")
    print(df.isnull().sum())

    # Check for date columns
    date_cols = []
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
            date_cols.append(col)
            if len(df[col].dropna()) > 0:
                try:
                    print(f"\nDate column '{col}' range: {df[col].min()} to {df[col].max()}")
                except:
                    print(f"\nDate column '{col}' has mixed types, cannot compute range")

    # Check for trade-related columns
    trade_indicators = ['buy', 'sell', 'long', 'short', 'trade', 'position', 'quantity', 'price', 'symbol', 'ticker', 'asset']
    found_cols = []
    for indicator in trade_indicators:
        for col in df.columns:
            if indicator in col.lower() and col not in found_cols:
                found_cols.append(col)

    if found_cols:
        print(f"\nTrade-related columns found: {found_cols}")
        for col in found_cols:
            if df[col].dtype in ['object', 'int64', 'float64']:
                print(f"\n'{col}' unique values: {df[col].nunique()}")
                if df[col].nunique() < 50:
                    print(f"Values: {df[col].value_counts().to_dict()}")

    print("\n" + "=" * 80)

print("\nAnalysis complete!")
