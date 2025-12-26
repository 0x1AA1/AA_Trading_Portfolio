#!/usr/bin/env python3
"""
Commodity Correlation Trading System
Professional-grade quantitative research framework
GPU/CPU accelerated ML for position sizing and direction
Based on "Advances in Financial Machine Learning" methodology
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd

# Check for GPU
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[INFO] GPU (CUDA) detected and will be used for acceleration")
except ImportError:
    GPU_AVAILABLE = False
    print("[INFO] GPU not available, using CPU with multi-threading")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cct_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import modules (will create these)
try:
    from data_pipeline import DataPipeline
    from feature_engineering import FeatureEngine
    from labeling import TripleBarrierLabeler
    from ml_models import MetaLabelingEngine
    from backtesting import EventDrivenBacktester
    from visualization import VisualizationSuite
except ImportError as e:
    logger.error(f"Missing module: {e}. Creating placeholder modules...")
    # Will create these modules next

def main():
    parser = argparse.ArgumentParser(
        description="CCT System - Commodity Correlation Trading with ML"
    )
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="today")
    parser.add_argument("--train-pct", type=float, default=0.7)
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU even if available")
    parser.add_argument("--n-jobs", type=int, default=-1, help="CPU cores to use")
    
    args = parser.parse_args()
    
    use_gpu = GPU_AVAILABLE and not args.no_gpu
    
    logger.info("="*70)
    logger.info("CCT SYSTEM - Commodity Correlation Trading")
    logger.info("="*70)
    logger.info(f"GPU Acceleration: {use_gpu}")
    logger.info(f"CPU Cores: {args.n_jobs if args.n_jobs > 0 else 'All available'}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info("="*70)
    
    # Pipeline execution
    logger.info("Step 1: Data Pipeline - Fetching prices...")
    pipeline = DataPipeline(use_gpu=use_gpu)
    data = pipeline.fetch_and_process(args.start, args.end)
    
    logger.info(f"Step 2: Feature Engineering...")
    feature_engine = FeatureEngine(use_gpu=use_gpu)
    features = feature_engine.generate_features(data)
    
    logger.info("Step 3: Triple Barrier Labeling...")
    labeler = TripleBarrierLabeler()
    labels = labeler.label_events(data, features)
    
    logger.info("Step 4: ML Model Training (MetaLabeling)...")
    ml_engine = MetaLabelingEngine(use_gpu=use_gpu, n_jobs=args.n_jobs)
    models = ml_engine.train(features, labels, train_split=args.train_pct)
    
    logger.info("Step 5: Backtesting...")
    backtester = EventDrivenBacktester()
    results = backtester.run(data, models)
    
    logger.info("Step 6: Visualization...")
    viz = VisualizationSuite()
    viz.generate_all_plots(data, features, models, results)
    
    logger.info("="*70)
    logger.info("EXECUTION COMPLETE")
    logger.info(f"Results saved to: results/")
    logger.info("="*70)

if __name__ == "__main__":
    main()
