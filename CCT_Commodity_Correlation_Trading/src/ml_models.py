"""
ML Models - MetaLabeling with ensemble methods
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class MetaLabelingEngine:
    def __init__(self, use_gpu=False, n_jobs=-1):
        self.use_gpu = use_gpu
        self.n_jobs = n_jobs
        self.models = {}
        
    def train(self, features, labels, train_split=0.7):
        """Train ML models"""
        logger.info("Training ML models...")
        
        for pair_name in features.keys():
            feat_df = features[pair_name]
            label_df = labels[pair_name]
            
            # Align
            common_idx = feat_df.index.intersection(label_df.index)
            X = feat_df.loc[common_idx].dropna()
            y = label_df.loc[X.index]['label']
            
            # Split
            split_idx = int(len(X) * train_split)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Direction classifier
            clf = RandomForestClassifier(n_estimators=100, n_jobs=self.n_jobs, random_state=42)
            clf.fit(X_train, y_train)
            
            # Size regressor (based on z-score magnitude)
            size_target = np.abs(X_train['com_zscore'])
            reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
            reg.fit(X_train, size_target)
            
            # Store
            self.models[pair_name] = {
                'direction_clf': clf,
                'size_reg': reg,
                'train_score': clf.score(X_train, y_train),
                'test_score': clf.score(X_test, y_test),
            }
            
            logger.info(f"{pair_name}: Train={self.models[pair_name]['train_score']:.3f}, Test={self.models[pair_name]['test_score']:.3f}")
        
        return self.models
