"""
Visualization Suite - Professional plots
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
sns.set_style("darkgrid")

class VisualizationSuite:
    def __init__(self):
        self.output_dir = Path(__file__).parent.parent / "results" / "plots"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def generate_all_plots(self, data, features, models, results):
        """Generate all visualizations"""
        logger.info("Generating plots...")
        
        # 1. Correlation heatmap
        self._plot_correlations(features)
        
        # 2. Feature importance
        self._plot_feature_importance(models)
        
        # 3. Model performance
        self._plot_model_performance(models)
        
        logger.info(f"Plots saved to {self.output_dir}")
        
    def _plot_correlations(self, features):
        """Correlation heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get first pair
        first_pair = list(features.values())[0]
        corr_cols = [c for c in first_pair.columns if 'corr' in c]
        
        if corr_cols:
            corr_data = first_pair[corr_cols].corr()
            sns.heatmap(corr_data, annot=True, cmap='RdYlGn', center=0, ax=ax)
            plt.title("Lead/Lag Correlation Matrix")
            plt.tight_layout()
            plt.savefig(self.output_dir / "correlation_heatmap.png", dpi=150)
            plt.close()
        
    def _plot_feature_importance(self, models):
        """Feature importance"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (pair_name, model) in enumerate(list(models.items())[:4]):
            clf = model['direction_clf']
            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            axes[idx].bar(range(len(importances)), importances[indices])
            axes[idx].set_title(f"{pair_name} - Feature Importance")
            axes[idx].set_xlabel("Feature Index")
            axes[idx].set_ylabel("Importance")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_importance.png", dpi=150)
        plt.close()
        
    def _plot_model_performance(self, models):
        """Model accuracy"""
        pair_names = list(models.keys())
        train_scores = [m['train_score'] for m in models.values()]
        test_scores = [m['test_score'] for m in models.values()]
        
        x = np.arange(len(pair_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
        ax.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
        
        ax.set_ylabel('Accuracy')
        ax.set_title('ML Model Performance by Pair')
        ax.set_xticks(x)
        ax.set_xticklabels(pair_names, rotation=45)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_performance.png", dpi=150)
        plt.close()
