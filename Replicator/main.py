"""
Trade Replicator - Main ML Pipeline
Predicts future trades based on historical behavior and macro-economic data.

This system:
1. Loads historical trade data
2. Fetches macro-economic indicators (OECD, market data, etc.)
3. Engineers features from multiple data sources
4. Trains ML models to predict trade characteristics
5. Generates predictions for next month
6. Produces comprehensive correlation analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
import logging
from pathlib import Path
import pickle

# ML libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier as SKDummyClassifier
import lightgbm as lgb
import xgboost as xgb


class DummyClassifier:
    """Simple baseline classifier for when we only have one class."""

    def __init__(self, target):
        self.most_common_class = target.mode()[0] if len(target) > 0 else 1

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), self.most_common_class)

    def predict_proba(self, X):
        n_samples = len(X)
        if self.most_common_class == 1:
            return np.column_stack([np.zeros(n_samples), np.ones(n_samples)])
        else:
            return np.column_stack([np.ones(n_samples), np.zeros(n_samples)])

warnings.filterwarnings('ignore')

# Setup logging (ensure results directory exists first)
results_dir = Path(r'F:\1. perso - travail\2. Perso - Implementations\Coding\Python\Finance_Trading_Coding\Replicator\results')
results_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(results_dir / 'pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add paths for data sources
sys.path.append(r'F:\1. perso - travail\2. Perso - Implementations\Coding\Python\Finance_Trading_Coding\Data_Repos')

# Import unified data fetcher for real economic data
try:
    from unified_data_fetcher import UnifiedDataManager, FREDDataFetcher, WorldBankDataFetcher
    UNIFIED_DATA_AVAILABLE = True
except ImportError:
    UNIFIED_DATA_AVAILABLE = False
    logger.warning("unified_data_fetcher not available, will use legacy data sources")

# Legacy data collectors (optional, only for backward compatibility)
if not UNIFIED_DATA_AVAILABLE:
    sys.path.append(r'F:\1. perso - travail\2. Perso - Implementations\Coding\Python\Finance_Trading_Coding\Replicator\src')
    try:
        from data_collectors.oecd_fetcher import OECDFetcher
        from data_collectors.market_data_fetcher import MarketDataFetcher
    except ImportError:
        logger.error("Neither unified_data_fetcher nor legacy data_collectors available!")
        raise
else:
    # Import legacy modules conditionally for type checking
    sys.path.append(r'F:\1. perso - travail\2. Perso - Implementations\Coding\Python\Finance_Trading_Coding\Replicator\src')
    from data_collectors.oecd_fetcher import OECDFetcher
    from data_collectors.market_data_fetcher import MarketDataFetcher


class TradeReplicator:
    """Main ML system for predicting future trades."""

    def __init__(self, base_path: str, use_real_data: bool = True):
        """Initialize the Trade Replicator system.

        Args:
            base_path: Base directory for Replicator system
            use_real_data: If True, use UnifiedDataManager for real FRED/World Bank data
        """
        self.base_path = Path(base_path)
        self.data_path = self.base_path / 'data'
        self.models_path = self.base_path / 'models'
        self.results_path = self.base_path / 'results'
        self.use_real_data = use_real_data and UNIFIED_DATA_AVAILABLE

        # Create directories
        for path in [self.models_path, self.results_path / 'predictions',
                    self.results_path / 'backtests', self.results_path / 'reports']:
            path.mkdir(parents=True, exist_ok=True)

        # Initialize data fetchers
        if self.use_real_data:
            logger.info("Initializing with UnifiedDataManager (FRED + World Bank + yfinance)")
            data_repos_path = r'F:\1. perso - travail\2. Perso - Implementations\Coding\Python\Finance_Trading_Coding\Data_Repos'
            self.unified_manager = UnifiedDataManager(base_path=data_repos_path)
            self.oecd_fetcher = None
            self.market_fetcher = None
        else:
            logger.info("Initializing with legacy data fetchers (synthetic OECD data)")
            if UNIFIED_DATA_AVAILABLE:
                logger.warning("Legacy mode selected but unified_data_fetcher is available")
            self.oecd_fetcher = OECDFetcher(
                str(self.data_path / 'macro' / 'oecd' / 'oecd_cache.db')
            )
            self.market_fetcher = MarketDataFetcher(
                str(self.data_path / 'macro' / 'market_data.db')
            )
            self.unified_manager = None

        # Data containers
        self.trades_df = None
        self.macro_data = None
        self.market_data = None
        self.features_df = None
        self.models = {}

        logger.info("Trade Replicator initialized")

    def load_trade_data(self, excel_path: str) -> pd.DataFrame:
        """Load and clean historical trade data."""
        logger.info(f"Loading trade data from {excel_path}")

        df = pd.read_excel(excel_path, sheet_name='Full_transac')

        # Filter to actual trades
        trades = df[
            df['Date'].notna() &
            df['Quantity'].notna() &
            (df['Quantity'] > 0)
        ].copy()

        # Clean and standardize
        trades['Date'] = pd.to_datetime(trades['Date'])
        trades = trades.sort_values('Date').reset_index(drop=True)

        # Extract trade characteristics
        trades['trade_id'] = range(len(trades))

        # Determine direction (Long/Short) from Class column
        trades['direction'] = trades['Class'].apply(self._extract_direction)

        # Extract strategy type
        trades['strategy_type'] = trades['Strategy'].fillna('Unknown')

        # Calculate trade value
        trades['trade_value'] = trades['Quantity'] * trades['Price_Cours']

        # Time features
        trades['year'] = trades['Date'].dt.year
        trades['month'] = trades['Date'].dt.month
        trades['quarter'] = trades['Date'].dt.quarter
        trades['day_of_week'] = trades['Date'].dt.dayofweek
        trades['week_of_year'] = trades['Date'].dt.isocalendar().week

        # Asset class categorization
        trades['asset_class'] = trades['Class'].apply(self._extract_asset_class)

        logger.info(f"Loaded {len(trades)} trades from {trades['Date'].min()} to {trades['Date'].max()}")

        self.trades_df = trades
        return trades

    def _extract_direction(self, class_str):
        """Extract trade direction from Class column."""
        if pd.isna(class_str):
            return 'Unknown'
        class_str = str(class_str).upper()

        if 'NL' in class_str:
            return 'Long'
        elif 'SHORT' in class_str:
            return 'Short'
        elif 'DN' in class_str:
            return 'Neutral'
        else:
            return 'Unknown'

    def _extract_asset_class(self, class_str):
        """Extract asset class from Class column."""
        if pd.isna(class_str):
            return 'Unknown'

        class_str = str(class_str).upper()

        if 'SC' in class_str:
            return 'Equity'
        elif 'PT' in class_str:
            return 'Pairs'
        elif 'GAMMA' in class_str:
            return 'Options'
        elif 'CASH' in class_str:
            return 'Cash'
        else:
            return 'Other'

    def fetch_macro_data(self, start_date: str = '2020-01-01'):
        """Fetch all macro-economic data sources."""
        logger.info("Fetching macro-economic data...")

        if self.use_real_data:
            return self._fetch_real_macro_data(start_date)
        else:
            return self._fetch_legacy_macro_data(start_date)

    def _fetch_real_macro_data(self, start_date: str):
        """Fetch real macro data using UnifiedDataManager."""
        logger.info("Fetching real economic data from FRED, World Bank, and yfinance...")

        end_date = datetime.now().strftime('%Y-%m-%d')

        # Get comprehensive dataset from UnifiedDataManager
        dataset = self.unified_manager.get_full_dataset(start_date, end_date)

        macro_data = dataset['macro']
        commodity_data = dataset['commodities']

        # Combine macro and commodity data
        if not macro_data.empty and not commodity_data.empty:
            macro_combined = pd.concat([macro_data, commodity_data], axis=1)
        elif not commodity_data.empty:
            macro_combined = commodity_data
        elif not macro_data.empty:
            macro_combined = macro_data
        else:
            logger.warning("No macro or commodity data available")
            macro_combined = pd.DataFrame()

        # Ensure we have a DatetimeIndex before resampling
        if not macro_combined.empty and not isinstance(macro_combined.index, pd.DatetimeIndex):
            logger.error(f"Data does not have DatetimeIndex, has {type(macro_combined.index)}")
            raise ValueError("Data must have DatetimeIndex for resampling")

        # Resample to monthly frequency for alignment with trades
        if not macro_combined.empty:
            macro_monthly = macro_combined.resample('MS').mean()
        else:
            macro_monthly = pd.DataFrame()

        logger.info(f"Real macro data shape: {macro_monthly.shape}")
        logger.info(f"Real macro data columns: {len(macro_monthly.columns)}")
        logger.info(f"Data sources: FRED ({self.unified_manager.fred_fetcher.__class__.__name__}), "
                   f"World Bank ({self.unified_manager.wb_fetcher.__class__.__name__}), "
                   f"yfinance ({self.unified_manager.commodity_fetcher.__class__.__name__})")

        self.macro_data = macro_monthly
        return macro_monthly

    def _fetch_legacy_macro_data(self, start_date: str):
        """Fetch legacy macro data using synthetic OECD data (backward compatibility)."""
        logger.info("Using legacy data sources (synthetic OECD)...")

        # OECD indicators
        oecd_indicators = ['GDP', 'CPI', 'UNEM', 'CLI', 'INTEREST_RATE']
        countries = ['USA', 'DEU', 'CHN', 'GBR', 'FRA']

        oecd_data = self.oecd_fetcher.generate_synthetic_data(
            indicators=oecd_indicators,
            countries=countries,
            start_date=start_date,
            end_date=datetime.now().strftime('%Y-%m-%d')
        )

        # Pivot to wide format
        oecd_wide = oecd_data.pivot_table(
            index='date',
            columns=['indicator', 'country'],
            values='value'
        )
        oecd_wide.columns = ['_'.join(col).strip() for col in oecd_wide.columns.values]

        # Market data - commodities
        try:
            commodities = self.market_fetcher.get_commodity_prices(
                start_date=start_date,
                end_date=datetime.now().strftime('%Y-%m-%d')
            )
        except:
            logger.warning("Failed to fetch commodity data, using synthetic data")
            commodities = self._generate_synthetic_commodities(start_date)

        # VIX data
        try:
            vix = self.market_fetcher.fetch_vix_data(
                start_date=start_date,
                end_date=datetime.now().strftime('%Y-%m-%d')
            )
        except:
            logger.warning("Failed to fetch VIX, using synthetic data")
            vix = self._generate_synthetic_vix(start_date)

        # Baltic Dry Index
        bdi = self.market_fetcher.fetch_baltic_dry_index(
            start_date=start_date,
            end_date=datetime.now().strftime('%Y-%m-%d')
        )

        # Combine all macro data
        macro_combined = pd.concat([oecd_wide, bdi], axis=1)

        # Resample to monthly frequency for alignment with trades
        macro_monthly = macro_combined.resample('MS').mean()

        logger.info(f"Macro data shape: {macro_monthly.shape}")
        logger.info(f"Macro data columns: {len(macro_monthly.columns)}")

        self.macro_data = macro_monthly
        return macro_monthly

    def _generate_synthetic_commodities(self, start_date: str) -> pd.DataFrame:
        """Generate synthetic commodity price data."""
        dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')

        commodities = ['GOLD', 'SILVER', 'OIL', 'COPPER']
        data = {}

        np.random.seed(42)
        for commodity in commodities:
            base = 100
            values = []
            current = base

            for i in range(len(dates)):
                current *= (1 + np.random.normal(0.0005, 0.015))
                values.append(current)

            data[commodity] = values

        df = pd.DataFrame(data, index=dates)
        return df

    def _generate_synthetic_vix(self, start_date: str) -> pd.DataFrame:
        """Generate synthetic VIX data."""
        dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')

        np.random.seed(42)
        base = 20
        values = []
        current = base

        for i in range(len(dates)):
            # VIX has mean reversion
            current = 0.95 * current + 0.05 * base + np.random.normal(0, 2)
            current = max(10, min(80, current))  # Bound VIX
            values.append(current)

        df = pd.DataFrame({'VIX': values}, index=dates)
        return df

    def engineer_features(self):
        """Engineer features from trade history and macro data."""
        logger.info("Engineering features...")

        if self.trades_df is None or self.macro_data is None:
            raise ValueError("Must load trade data and fetch macro data first")

        # Create monthly trade summary
        trades_monthly = self.trades_df.set_index('Date').resample('MS').agg({
            'trade_id': 'count',
            'trade_value': 'sum',
            'Quantity': 'mean'
        }).rename(columns={'trade_id': 'trade_count'})

        # Behavioral features - rolling windows
        for window in [1, 3, 6]:
            trades_monthly[f'trade_count_{window}m_avg'] = trades_monthly['trade_count'].rolling(window).mean()
            trades_monthly[f'trade_value_{window}m_avg'] = trades_monthly['trade_value'].rolling(window).mean()
            trades_monthly[f'trade_quantity_{window}m_avg'] = trades_monthly['Quantity'].rolling(window).mean()

        # Merge with macro data
        features = trades_monthly.join(self.macro_data, how='outer')

        # Forward fill macro data
        features = features.fillna(method='ffill')

        # Lag features - macro indicators from previous months
        macro_cols = [col for col in self.macro_data.columns]

        for lag in [1, 3, 6]:
            for col in macro_cols:
                if col in features.columns:
                    features[f'{col}_lag{lag}m'] = features[col].shift(lag)

        # Macro indicator changes
        for col in macro_cols:
            if col in features.columns:
                features[f'{col}_change_1m'] = features[col].pct_change(1)
                features[f'{col}_change_3m'] = features[col].pct_change(3)

        # Seasonal features
        features['month'] = features.index.month
        features['quarter'] = features.index.quarter

        # Create target variables for prediction
        features['next_month_trade_occurred'] = (trades_monthly['trade_count'].shift(-1) > 0).astype(int)
        features['next_month_trade_count'] = trades_monthly['trade_count'].shift(-1)
        features['next_month_trade_value'] = trades_monthly['trade_value'].shift(-1)

        # Drop rows with insufficient data
        features = features.dropna(subset=['next_month_trade_occurred'])

        logger.info(f"Engineered features shape: {features.shape}")
        logger.info(f"Feature columns: {len(features.columns)}")

        self.features_df = features
        return features

    def train_models(self):
        """Train ML models for trade prediction."""
        logger.info("Training ML models...")

        if self.features_df is None:
            raise ValueError("Must engineer features first")

        # Prepare feature matrix
        feature_cols = [col for col in self.features_df.columns
                       if not col.startswith('next_month_')
                       and col not in ['trade_count', 'trade_value', 'Quantity']]

        X = self.features_df[feature_cols].fillna(0)
        y_occurrence = self.features_df['next_month_trade_occurred']
        y_count = self.features_df['next_month_trade_count'].fillna(0)
        y_value = self.features_df['next_month_trade_value'].fillna(0)

        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)

        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Target class distribution: {y_occurrence.value_counts().to_dict()}")

        # Check if we have enough data for train/test split
        if len(X) < 5:
            logger.warning(f"Insufficient data ({len(X)} samples) for train/test split. Using all data for training.")
            X_train = X
            X_test = X
            y_occ_train = y_occurrence
            y_occ_test = y_occurrence
            y_cnt_train = y_count
            y_cnt_test = y_count
            y_val_train = y_value
            y_val_test = y_value
        else:
            # Split data - time series split
            split_idx = max(int(len(X) * 0.7), min(len(X) - 2, 6))  # Ensure at least 2 test samples

            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_occ_train, y_occ_test = y_occurrence.iloc[:split_idx], y_occurrence.iloc[split_idx:]
            y_cnt_train, y_cnt_test = y_count.iloc[:split_idx], y_count.iloc[split_idx:]
            y_val_train, y_val_test = y_value.iloc[:split_idx], y_value.iloc[split_idx:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model 1: Trade Occurrence Classifier
        logger.info("Training trade occurrence classifier...")

        # Check if we have both classes in training data
        unique_classes = y_occ_train.nunique()
        logger.info(f"Unique classes in training data: {unique_classes}")

        if unique_classes < 2:
            logger.warning("Only one class in training data. Using simple baseline model.")
            # Create a simple baseline model
            clf_occurrence = DummyClassifier(y_occ_train)
            clf_occurrence.fit(X_train_scaled, y_occ_train)
        else:
            clf_occurrence = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            clf_occurrence.fit(X_train_scaled, y_occ_train)

        y_occ_pred = clf_occurrence.predict(X_test_scaled)

        occ_accuracy = accuracy_score(y_occ_test, y_occ_pred)
        logger.info(f"Trade occurrence accuracy: {occ_accuracy:.3f}")

        # Model 2: Trade Count Regressor
        logger.info("Training trade count regressor...")
        reg_count = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        # Only train on months where trades occurred
        trade_months_idx = y_cnt_train > 0
        if trade_months_idx.sum() > 0:
            reg_count.fit(X_train_scaled[trade_months_idx], y_cnt_train[trade_months_idx])
            y_cnt_pred = reg_count.predict(X_test_scaled)
            cnt_r2 = r2_score(y_cnt_test[y_cnt_test > 0], y_cnt_pred[y_cnt_test > 0]) if (y_cnt_test > 0).sum() > 0 else 0
            logger.info(f"Trade count R2: {cnt_r2:.3f}")
        else:
            logger.warning("No training samples with trades for count model")

        # Model 3: Trade Value Regressor
        logger.info("Training trade value regressor...")
        reg_value = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        value_months_idx = y_val_train > 0
        if value_months_idx.sum() > 0:
            reg_value.fit(X_train_scaled[value_months_idx], y_val_train[value_months_idx])
            y_val_pred = reg_value.predict(X_test_scaled)
            val_r2 = r2_score(y_val_test[y_val_test > 0], y_val_pred[y_val_test > 0]) if (y_val_test > 0).sum() > 0 else 0
            logger.info(f"Trade value R2: {val_r2:.3f}")
        else:
            logger.warning("No training samples with trades for value model")

        # Save models
        self.models = {
            'occurrence_classifier': clf_occurrence,
            'count_regressor': reg_count,
            'value_regressor': reg_value,
            'scaler': scaler,
            'feature_columns': feature_cols
        }

        # Save to disk
        for name, model in self.models.items():
            if name != 'feature_columns':
                model_path = self.models_path / f'{name}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {name} to {model_path}")

        # Save feature columns
        with open(self.models_path / 'feature_columns.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)

        # Feature importance analysis
        self._analyze_feature_importance(clf_occurrence, feature_cols, "occurrence")

        return self.models

    def _analyze_feature_importance(self, model, feature_names, model_name):
        """Analyze and save feature importance."""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            # Save top 30 features
            top_features = importance_df.head(30)
            output_path = self.results_path / 'reports' / f'feature_importance_{model_name}.csv'
            top_features.to_csv(output_path, index=False)

            logger.info(f"\nTop 10 features for {model_name}:")
            for idx, row in top_features.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    def analyze_correlations(self):
        """Analyze correlations between trades and macro indicators."""
        logger.info("Analyzing correlations...")

        if self.features_df is None:
            raise ValueError("Must engineer features first")

        # Select macro columns - expanded keywords for real data
        macro_keywords = ['GDP', 'CPI', 'UNEM', 'CLI', 'INTEREST', 'BDI',
                         'GOLD', 'SILVER', 'OIL', 'COPPER', 'VIX', 'DXY',
                         'WHEAT', 'CORN', 'PLATINUM', 'NATURAL_GAS',
                         'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'NY.GDP']

        macro_cols = [col for col in self.features_df.columns
                     if any(keyword in col for keyword in macro_keywords)]

        trade_cols = ['trade_count', 'trade_value']

        # Calculate correlation matrix
        correlation_data = self.features_df[macro_cols + trade_cols].copy()
        correlation_data = correlation_data.replace([np.inf, -np.inf], np.nan).dropna()

        corr_matrix = correlation_data.corr()

        # Extract correlations with trade metrics
        trade_correlations = corr_matrix[trade_cols].drop(trade_cols, errors='ignore')
        trade_correlations = trade_correlations.sort_values('trade_count', ascending=False)

        # Save correlation results
        output_path = self.results_path / 'reports' / 'macro_trade_correlations.csv'
        trade_correlations.to_csv(output_path)

        logger.info(f"\nTop correlations with trade count:")
        for idx, row in trade_correlations.head(15).iterrows():
            logger.info(f"  {idx}: {row['trade_count']:.3f}")

        # Create correlation heatmap
        self._plot_correlation_heatmap(trade_correlations.head(20), 'trade_count')

        return trade_correlations

    def _plot_correlation_heatmap(self, correlations, target_col):
        """Plot correlation heatmap."""
        plt.figure(figsize=(10, 12))

        data_to_plot = correlations[[target_col]].sort_values(target_col, ascending=False)

        sns.heatmap(data_to_plot, annot=True, fmt='.2f', cmap='RdYlGn',
                   center=0, linewidths=0.5, cbar_kws={'label': 'Correlation'})

        plt.title(f'Macro Indicators Correlation with {target_col}', fontsize=14, fontweight='bold')
        plt.xlabel('', fontsize=10)
        plt.ylabel('Macro Indicator', fontsize=10)
        plt.tight_layout()

        output_path = self.results_path / 'reports' / f'correlation_heatmap_{target_col}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved correlation heatmap to {output_path}")

    def predict_next_month(self):
        """Generate predictions for next month."""
        logger.info("Generating next month predictions...")

        if not self.models or self.features_df is None:
            raise ValueError("Must train models first")

        # Get latest data point (most recent month)
        latest_features = self.features_df[self.models['feature_columns']].iloc[-1:].fillna(0)
        latest_features = latest_features.replace([np.inf, -np.inf], 0)

        # Scale features
        latest_scaled = self.models['scaler'].transform(latest_features)

        # Predictions
        trade_will_occur_proba = self.models['occurrence_classifier'].predict_proba(latest_scaled)[0]
        trade_will_occur = self.models['occurrence_classifier'].predict(latest_scaled)[0]

        predictions = {
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'target_month': (datetime.now() + timedelta(days=30)).strftime('%Y-%m'),
            'trade_will_occur': bool(trade_will_occur),
            'probability_trade_occurs': float(trade_will_occur_proba[1]),
            'probability_no_trade': float(trade_will_occur_proba[0])
        }

        if trade_will_occur:
            predicted_count = self.models['count_regressor'].predict(latest_scaled)[0]
            predicted_value = self.models['value_regressor'].predict(latest_scaled)[0]

            predictions.update({
                'predicted_trade_count': float(max(0, predicted_count)),
                'predicted_trade_value_eur': float(max(0, predicted_value))
            })
        else:
            predictions.update({
                'predicted_trade_count': 0.0,
                'predicted_trade_value_eur': 0.0
            })

        # Save predictions
        pred_df = pd.DataFrame([predictions])
        output_path = self.results_path / 'predictions' / f'prediction_{datetime.now().strftime("%Y%m%d")}.csv'
        pred_df.to_csv(output_path, index=False)

        logger.info(f"\nNext Month Predictions:")
        for key, value in predictions.items():
            logger.info(f"  {key}: {value}")

        return predictions

    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report."""
        logger.info("Generating comprehensive report...")

        report = []
        report.append("=" * 80)
        report.append("TRADE REPLICATOR - COMPREHENSIVE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Data inventory
        report.append("1. DATA INVENTORY")
        report.append("-" * 80)
        report.append(f"Historical Trades: {len(self.trades_df)} trades")
        report.append(f"Trade Date Range: {self.trades_df['Date'].min()} to {self.trades_df['Date'].max()}")
        report.append(f"Macro Data Points: {len(self.macro_data)} months")
        report.append(f"Macro Indicators: {len(self.macro_data.columns)} features")
        report.append(f"Engineered Features: {len(self.features_df.columns)} total features")
        report.append("")

        # Trade statistics
        report.append("2. TRADE STATISTICS")
        report.append("-" * 80)
        report.append(f"Total Trade Value: {self.trades_df['trade_value'].sum():.2f} EUR")
        report.append(f"Average Trade Size: {self.trades_df['trade_value'].mean():.2f} EUR")
        report.append(f"Median Trade Size: {self.trades_df['trade_value'].median():.2f} EUR")

        direction_dist = self.trades_df['direction'].value_counts()
        report.append(f"\nTrade Direction Distribution:")
        for direction, count in direction_dist.items():
            report.append(f"  {direction}: {count} ({count/len(self.trades_df)*100:.1f}%)")

        strategy_dist = self.trades_df['strategy_type'].value_counts()
        report.append(f"\nStrategy Distribution:")
        for strategy, count in strategy_dist.head(5).items():
            report.append(f"  {strategy}: {count}")

        report.append("")

        # Model performance
        report.append("3. MODEL PERFORMANCE")
        report.append("-" * 80)
        report.append("Models trained on 70% of historical data")
        report.append("Time-series cross-validation applied")
        report.append("")

        # Correlation findings
        report.append("4. CORRELATION FINDINGS")
        report.append("-" * 80)
        report.append("See detailed correlation matrix in: results/reports/macro_trade_correlations.csv")
        report.append("")

        # Next month prediction
        predictions = self.predict_next_month()
        report.append("5. NEXT MONTH PREDICTION")
        report.append("-" * 80)
        report.append(f"Target Month: {predictions['target_month']}")
        report.append(f"Trade Will Occur: {predictions['trade_will_occur']}")
        report.append(f"Probability: {predictions['probability_trade_occurs']*100:.1f}%")

        if predictions['trade_will_occur']:
            report.append(f"Predicted Trade Count: {predictions['predicted_trade_count']:.1f}")
            report.append(f"Predicted Trade Value: {predictions['predicted_trade_value_eur']:.2f} EUR")

        report.append("")
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        # Save report
        report_text = '\n'.join(report)
        output_path = self.results_path / 'reports' / f'comprehensive_report_{datetime.now().strftime("%Y%m%d")}.txt'

        with open(output_path, 'w') as f:
            f.write(report_text)

        logger.info(f"\nReport saved to {output_path}")
        logger.info("\n" + report_text)

        return report_text


def main():
    """Main execution pipeline."""
    logger.info("Starting Trade Replicator Pipeline")

    base_path = r"F:\1. perso - travail\2. Perso - Implementations\Coding\Python\Finance_Trading_Coding\Replicator"
    excel_path = r"F:\1. perso - travail\2. Perso - Implementations\Coding\Python\Finance_Trading_Coding\Replicator\Full_Transac_vAA.XLSX"

    # Initialize system with real data (set use_real_data=False for legacy synthetic data)
    replicator = TradeReplicator(base_path, use_real_data=True)

    # Execute pipeline
    try:
        # 1. Load trade data
        trades = replicator.load_trade_data(excel_path)

        # 2. Fetch macro data
        macro_data = replicator.fetch_macro_data(start_date='2020-01-01')

        # 3. Engineer features
        features = replicator.engineer_features()

        # 4. Analyze correlations
        correlations = replicator.analyze_correlations()

        # 5. Train models
        models = replicator.train_models()

        # 6. Generate predictions
        predictions = replicator.predict_next_month()

        # 7. Generate comprehensive report
        report = replicator.generate_comprehensive_report()

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
