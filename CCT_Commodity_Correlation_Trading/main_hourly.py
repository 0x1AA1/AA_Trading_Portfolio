# -*- coding: utf-8 -*-
"""
CCT Hourly Correlation Analyzer
Main entry point for hourly commodity-equity correlation analysis

Usage:
    python main_hourly.py
"""
import sys
from pathlib import Path

# Add parent directory to path for utils
sys.path.append(str(Path(__file__).parent.parent))

# Run the analyzer
exec(open(Path(__file__).parent / 'analyzer.py').read())
