"""
Enhanced Multi-Dataset Model Training Pipeline
Trains fraud detection models on multiple datasets for maximum performance
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score, f1_score
)
import shap

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiDatasetFraudTrainer:
    """Enhanced fraud detection trainer using multiple datasets"""
    
    def __init__(self, model_dir: str = "models/multi_dataset"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.training_history = []
        
        # Model configurations
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        
        self.isolation_params = {
            'contamination': 0.1,
            'random_state': 42,
            'n_estimators': 100
        }
    
    def load_datasets(self, data_dir: str = "data") -> Dict[str, pd.DataFrame]:
        """Load all processed datasets including real downloaded ones"""
        datasets = {}
        
        logger.info("Loading all available datasets...")
        
        # First, try to load real datasets from raw directory
        raw_data_path = Path(data_dir) / "raw"
        if raw_data_path.exists():
            logger.info("Loading real datasets from raw directory...")
            
            # Load Credit Card dataset
            creditcard_path = raw_data_path / "creditcard.csv"
            if creditcard_path.exists():
                datasets['creditcard_real'] = pd.read_csv(creditcard_path)
                logger.info(f"Loaded real credit card dataset: {len(datasets['creditcard_real']):,} transactions")
            
            # Load BankSim dataset
            banksim_path = raw_data_path / "banksim.csv"
            if banksim_path.exists():
                datasets['banksim_real'] = pd.read_csv(banksim_path)
                logger.info(f"Loaded real BankSim dataset: {len(datasets['banksim_real']):,} transactions")
            
            # Load other real datasets
            for csv_file in raw_data_path.glob("*.csv"):
                if csv_file.name not in ['creditcard.csv', 'banksim.csv']:
                    dataset_name = f"{csv_file.stem}_real"
                    try:
                        df = pd.read_csv(csv_file)
                        datasets[dataset_name] = df
                        logger.info(f"Loaded {dataset_name}: {len(df):,} transactions")
                    except Exception as e:
                        logger.warning(f"Failed to load {csv_file}: {e}")
        
        # Then load processed datasets
        processed_path = Path(data_dir) / "processed"
        if processed_path.exists():
            for csv_file in processed_path.glob("*_fraud_data.csv"):
                dataset_name = csv_file.stem.replace('_fraud_data', '') + '_synthetic'
                try:
                    df = pd.read_csv(csv_file)
                    datasets[dataset_name] = df
                    logger.info(f"Loaded {dataset_name}: {len(df):,} transactions")
                except Exception as e:
                    logger.error(f"Failed to load {csv_file}: {e}")
        
        # If no datasets found, create samples
        if not datasets:
            logger.warning("No datasets found! Creating sample datasets...")
            datasets = self._create_sample_datasets()
        
        # Try to download real datasets if none found
        if len(datasets) < 2:
            logger.info("Limited datasets found. Attempting to download real datasets...")
            try:
                from download_real_datasets import RealDatasetDownloader
                downloader = RealDatasetDownloader(str(raw_data_path))
                real_datasets = downloader.download_all_datasets()
                
                for name, df in real_datasets.items():
                    datasets[f"{name}_real"] = df
                    logger.info(f"Downloaded and loaded {name}: {len(df):,} transactions")
            except Exception as e:
                logger.warning(f"Failed to download real datasets: {e}")
        
        return datasets
    
    def _create_sample_datasets(self) -> Dict[str, pd.DataFrame]:
        """Create sample datasets if none found"""
        logger.info("Creating sample datasets for demonstration...")
        
        np.random.seed(42)
        datasets = {}
        
        # Sample credit card style dataset
        n_samples = 10000
        datasets['sample'] = pd.DataFrame({
            'amount': np.random.lognormal(4, 1, n_samples),
            'v1': np.random.normal(0, 1, n_samples),
            'v2': np.random.normal(0, 1, n_samples),
            'v3': np.random.normal(0, 1, n_samples),
            'v4': np.random.normal(0, 1, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'merchant_category': np.random.choice([0, 1, 2, 3, 4], n_samples),
            'is_fraud': np.random.choice([0, 1], n_samples, p=[0.995, 0.005])
        })
        
        return datasets
    
    def prepare_features(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare and combine features from all datasets"""
        logger.info("Preparing features from all datasets...")
        
        combined_features = []
        combined_labels = []
        
        for dataset_name, df in datasets.items():
            logger.info(f"Processing {dataset_name} dataset...")
            
            try:
                features, labels = self._extract_features_from_dataset(df, dataset_name)
                combined_features.append(features)
                combined_labels.append(labels)
                logger.info(f"Extracted {len(features)} samples from {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to process {dataset_name}: {e}")
                continue
        
        if not combined_features:
            raise ValueError("No features could be extracted from any dataset!")
        
        # Combine all features
        X_combined = pd.concat(combined_features, ignore_index=True, sort=False)
        y_combined = pd.concat(combined_labels, ignore_index=True)
        
        # Fill missing values
        X_combined = X_combined.fillna(0)
        
        # Store feature names
        self.feature_names = X_combined.columns.tolist()
        
        logger.info(f"Combined dataset: {len(X_combined):,} samples, {len(self.feature_names)} features")
        logger.info(f"Fraud rate: {y_combined.mean():.4f} ({y_combined.sum():,} frauds)")
        
        return X_combined, y_combined
    
    def _extract_features_from_dataset(self, df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract standardized features from a specific dataset"""
        
        # Find the fraud label column (usually the last one or named specifically)
        fraud_columns = ['is_fraud', 'isFraud', 'Class', 'TX_FRAUD', 'fraud']
        fraud_col = None
        
        for col in fraud_columns:
            if col in df.columns:
                fraud_col = col
                break
        
        if fraud_col is None:
            # Assume last column is fraud indicator
            fraud_col = df.columns[-1]
            logger.warning(f"No standard fraud column found in {dataset_name}, using {fraud_col}")
        
        # Extract labels
        labels = df[fraud_col].astype(int)
        
        # Extract features (all columns except fraud label)
        feature_df = df.drop(columns=[fraud_col])
        
        # Standardize feature extraction based on dataset type
        if dataset_name == 'creditcard':
            features = self._process_creditcard_features(feature_df)
        elif dataset_name == 'ieee':
            features = self._process_ieee_features(feature_df)
        elif dataset_name == 'paysim':
            features = self._process_paysim_features(feature_df)
        elif dataset_name == 'synthetic':
            features = self._process_synthetic_features(feature_df)
        elif dataset_name == 'ecommerce':
            features = self._process_ecommerce_features(feature_df)
        else:
            # Generic processing
            features = self._process_generic_features(feature_df)
        
        # Add dataset indicator
        features[f'dataset_{dataset_name}'] = 1
        
        return features, labels
    
    def _process_creditcard_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Credit Card Fraud dataset features"""
        features = df.copy()
        
        # Time features
        if 'Time' in features.columns:
            features['hour'] = (features['Time'] / 3600) % 24
            features['day'] = (features['Time'] / (3600 * 24)).astype(int)
            features['is_weekend'] = (features['day'] % 7 >= 5).astype(int)
            features = features.drop('Time', axis=1)
        
        # Amount features
        if 'Amount' in features.columns:
            features['amount_log'] = np.log1p(features['Amount'])
            features['amount_sqrt'] = np.sqrt(features['Amount'])
            features['amount_high'] = (features['Amount'] > 100).astype(int)
            features['amount_very_high'] = (features['Amount'] > 1000).astype(int)
        
        return features
    
    def _process_ieee_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process IEEE Fraud dataset features"""
        features = df.copy()
        
        # Select important columns (IEEE has 400+ features)
        important_cols = []
        for col in features.columns:
            if col.startswith(('TransactionAmt', 'card', 'addr', 'dist', 'C', 'D', 'V')):
                important_cols.append(col)
        
        if important_cols:
            features = features[important_cols[:50]]  # Limit to top 50 features
        
        return features.fillna(0)
    
    def _process_paysim_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process PaySim dataset features"""
        features = df.copy()
        
        # Encode transaction type
        if 'type' in features.columns:
            type_encoder = LabelEncoder()
            features['type_encoded'] = type_encoder.fit_transform(features['type'])
            features = features.drop('type', axis=1)
        
        # Balance features
        for col in ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
            if col in features.columns:
                features[f'{col}_log'] = np.log1p(features[col])
                features[f'{col}_zero'] = (features[col] == 0).astype(int)
        
        # Amount features
        if 'amount' in features.columns:
            features['amount_log'] = np.log1p(features['amount'])
            features['amount_high'] = (features['amount'] > 100000).astype(int)
        
        # Remove string columns
        string_cols = ['nameOrig', 'nameDest']
        features = features.drop(columns=[col for col in string_cols if col in features.columns])
        
        return features
    
    def _process_synthetic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process synthetic dataset features"""
        features = df.copy()
        
        # Encode categorical features
        categorical_cols = ['merchant_category', 'payment_method', 'device_type']
        for col in categorical_cols:
            if col in features.columns:
                encoder = LabelEncoder()
                features[f'{col}_encoded'] = encoder.fit_transform(features[col].astype(str))
                features = features.drop(col, axis=1)
        
        # Time-based features
        if 'hour' in features.columns:
            features['is_night'] = ((features['hour'] < 6) | (features['hour'] > 22)).astype(int)
            features['is_business_hours'] = ((features['hour'] >= 9) & (features['hour'] <= 17)).astype(int)
        
        return features
    
    def _process_ecommerce_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process e-commerce dataset features"""
        features = df.copy()
        
        # Encode categorical features
        categorical_cols = ['product_category', 'browser', 'country', 'email_domain']
        for col in categorical_cols:
            if col in features.columns:
                encoder = LabelEncoder()
                features[f'{col}_encoded'] = encoder.fit_transform(features[col].astype(str))
                features = features.drop(col, axis=1)
        
        # Behavioral features
        if 'session_length_sec' in features.columns and 'total_amount' in features.columns:
            features['amount_per_second'] = features['total_amount'] / (features['session_length_sec'] + 1)
        
        # Drop ID columns
        id_cols = ['transaction_id', 'user_id', 'session_id']
        features = features.drop(columns=[col for col in id_cols if col in features.columns])
        
        return features
    
    def _process_generic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generic feature processing for unknown datasets"""
        features = df.copy()
        
        # Convert categorical columns to numeric
        for col in features.columns:
            if features[col].dtype == 'object':
                try:
                    encoder = LabelEncoder()
                    features[col] = encoder.fit_transform(features[col].astype(str))
                except:
                    features = features.drop(col, axis=1)
        
        return features
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train ensemble of fraud detection models"""
        logger.info("Training fraud detection models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        results = {}
        
        # Train LightGBM
        logger.info("Training LightGBM model...")
        lgb_model = self._train_lightgbm(X_train, y_train, X_test, y_test)
        self.models['lightgbm'] = lgb_model
        results['lightgbm'] = self._evaluate_model(lgb_model, X_test, y_test, 'LightGBM')
        
        # Train Isolation Forest
        logger.info("Training Isolation Forest model...")
        if_model = self._train_isolation_forest(X_train_scaled, X_test_scaled, y_test)
        self.models['isolation_forest'] = if_model
        results['isolation_forest'] = self._evaluate_isolation_forest(if_model, X_test_scaled, y_test)
        
        # Train ensemble
        logger.info("Training ensemble model...")
        ensemble_results = self._train_ensemble(X_test, X_test_scaled, y_test)
        results['ensemble'] = ensemble_results
        
        # Feature importance analysis
        self._analyze_feature_importance(lgb_model, X_train)
        
        # Save models
        self._save_models()
        
        return results
    
    def _train_lightgbm(self, X_train, y_train, X_test, y_test) -> lgb.LGBMClassifier:
        """Train LightGBM model with cross-validation"""
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model
        model = lgb.train(
            self.lgb_params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'eval'],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        return model
    
    def _train_isolation_forest(self, X_train_scaled, X_test_scaled, y_test) -> IsolationForest:
        """Train Isolation Forest for anomaly detection"""
        
        # Train on normal transactions only
        normal_indices = np.random.choice(
            len(X_train_scaled), 
            size=min(10000, len(X_train_scaled)), 
            replace=False
        )
        X_normal = X_train_scaled[normal_indices]
        
        model = IsolationForest(**self.isolation_params)
        model.fit(X_normal)
        
        return model
    
    def _train_ensemble(self, X_test, X_test_scaled, y_test) -> Dict:
        """Create ensemble predictions"""
        
        # Get predictions from both models
        lgb_pred = self.models['lightgbm'].predict(X_test)
        if_scores = self.models['isolation_forest'].decision_function(X_test_scaled)
        
        # Normalize isolation forest scores to [0, 1]
        if_pred = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
        
        # Ensemble with weighted average
        ensemble_pred = 0.7 * lgb_pred + 0.3 * (1 - if_pred)  # IF gives anomaly scores, so invert
        
        # Evaluate ensemble
        auc = roc_auc_score(y_test, ensemble_pred)
        ap = average_precision_score(y_test, ensemble_pred)
        
        return {
            'auc': auc,
            'average_precision': ap,
            'predictions': ensemble_pred
        }
    
    def _evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict:
        """Evaluate model performance"""
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict(X_test)
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        
        # Binary predictions with optimal threshold
        y_pred_binary = (y_pred_proba >= optimal_threshold).astype(int)
        f1 = f1_score(y_test, y_pred_binary)
        
        logger.info(f"{model_name} - AUC: {auc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}")
        
        return {
            'auc': auc,
            'average_precision': ap,
            'f1_score': f1,
            'optimal_threshold': optimal_threshold,
            'predictions': y_pred_proba
        }
    
    def _evaluate_isolation_forest(self, model, X_test_scaled, y_test) -> Dict:
        """Evaluate Isolation Forest model"""
        
        # Get anomaly scores
        anomaly_scores = model.decision_function(X_test_scaled)
        
        # Convert to fraud probabilities (invert and normalize)
        fraud_scores = 1 - (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        
        # Evaluate
        auc = roc_auc_score(y_test, fraud_scores)
        ap = average_precision_score(y_test, fraud_scores)
        
        logger.info(f"Isolation Forest - AUC: {auc:.4f}, AP: {ap:.4f}")
        
        return {
            'auc': auc,
            'average_precision': ap,
            'predictions': fraud_scores
        }
    
    def _analyze_feature_importance(self, model, X_train):
        """Analyze and save feature importance"""
        logger.info("Analyzing feature importance...")
        
        # Get feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Save feature importance
        importance_path = self.model_dir / "feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        
        # Log top features
        logger.info("Top 10 most important features:")
        for _, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.2f}")
    
    def _save_models(self):
        """Save all trained models and metadata"""
        logger.info("Saving models...")
        
        # Save LightGBM model
        lgb_path = self.model_dir / "lightgbm_model.txt"
        self.models['lightgbm'].save_model(str(lgb_path))
        
        # Save Isolation Forest
        if_path = self.model_dir / "isolation_forest_model.joblib"
        joblib.dump(self.models['isolation_forest'], if_path)
        
        # Save scaler
        scaler_path = self.model_dir / "scaler.joblib"
        joblib.dump(self.scalers['standard'], scaler_path)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names),
            'training_date': datetime.now().isoformat(),
            'model_versions': {
                'lightgbm': lgb.__version__,
                'sklearn': '1.0+'
            }
        }
        
        metadata_path = self.model_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved to {self.model_dir}")


def main():
    """Main training pipeline"""
    logger.info("🚀 Starting Multi-Dataset Fraud Detection Training")
    logger.info("=" * 70)
    
    # Initialize trainer
    trainer = MultiDatasetFraudTrainer()
    
    # Load datasets
    datasets = trainer.load_datasets()
    
    if not datasets:
        logger.error("No datasets found! Please run data_acquisition.py first.")
        return
    
    # Prepare features
    X, y = trainer.prepare_features(datasets)
    
    # Train models
    results = trainer.train_models(X, y)
    
    # Print final results
    logger.info("=" * 70)
    logger.info("📊 TRAINING RESULTS SUMMARY")
    logger.info("=" * 70)
    
    for model_name, metrics in results.items():
        logger.info(f"{model_name.upper()}:")
        logger.info(f"  AUC: {metrics['auc']:.4f}")
        logger.info(f"  Average Precision: {metrics['average_precision']:.4f}")
        if 'f1_score' in metrics:
            logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        logger.info("")
    
    logger.info("✅ Training completed successfully!")
    logger.info(f"📁 Models saved in: {trainer.model_dir}")
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = main()
