"""
Feature Engineering - Technical features from price data
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class FeatureEngine:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        
    def generate_features(self, data):
        """Generate features for ML"""
        logger.info("Engineering features...")
        
        prices = data["prices"]
        returns = data["returns"]
        pairs = data["pairs"]
        
        features = {}
        
        for pair in pairs:
            eq = pair["equity"]
            com = pair["commodity"]
            
            df = pd.DataFrame()
            
            # Returns
            df['eq_ret'] = returns[eq]
            df['com_ret'] = returns[com]
            
            # Volatility
            df['eq_vol'] = returns[eq].rolling(21).std()
            df['com_vol'] = returns[com].rolling(21).std()
            
            # Z-scores
            df['com_zscore'] = (returns[com] - returns[com].rolling(60).mean()) / returns[com].rolling(60).std()
            
            # Lag correlations
            for lag in [1, 3, 5, 10]:
                df[f'corr_lag{lag}'] = returns[com].rolling(60).corr(returns[eq].shift(-lag))
            
            features[pair["name"]] = df.dropna()
        
        logger.info(f"Generated features for {len(pairs)} pairs")
        return features
