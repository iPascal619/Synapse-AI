import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import joblib
import json
from datetime import datetime

import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import shap
import onnx
import onnxmltools
from onnxmltools.convert import convert_lightgbm

from data_pipeline.schemas import FeatureVector


class FraudDetectionModelTrainer:
    """
    Trains both supervised and unsupervised models for fraud detection.
    
    Implements LightGBM for supervised classification and Isolation Forest
    for unsupervised anomaly detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model trainer with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.training_metrics = {}
        
    def load_training_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare training data.
        
        Args:
            data_path: Path to training data CSV file
            
        Returns:
            Tuple of (features_df, labels_df)
        """
        self.logger.info(f"Loading training data from {data_path}")
        
        # Load data (expecting CSV with features + is_fraud label)
        data = pd.read_csv(data_path)
        
        # Separate features and labels
        feature_columns = [col for col in data.columns if col not in ['transaction_id', 'is_fraud']]
        features = data[feature_columns]
        labels = data['is_fraud'] if 'is_fraud' in data.columns else None
        
        self.feature_names = feature_columns
        
        self.logger.info(f"Loaded {len(data)} samples with {len(feature_columns)} features")
        if labels is not None:
            fraud_rate = labels.mean()
            self.logger.info(f"Fraud rate: {fraud_rate:.3f} ({labels.sum()} fraud cases)")
        
        return features, labels
    
    def preprocess_features(self, features: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Preprocess features for model training.
        
        Args:
            features: Raw feature DataFrame
            fit_scaler: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Preprocessed features
        """
        # Handle missing values
        features = features.fillna(0)
        
        # Scale numerical features for Isolation Forest
        if fit_scaler:
            self.scalers['standard'] = StandardScaler()
            features_scaled = pd.DataFrame(
                self.scalers['standard'].fit_transform(features),
                columns=features.columns,
                index=features.index
            )
        else:
            features_scaled = pd.DataFrame(
                self.scalers['standard'].transform(features),
                columns=features.columns,
                index=features.index
            )
        
        return features_scaled
    
    def train_supervised_model(self, features: pd.DataFrame, labels: pd.Series) -> Dict[str, Any]:
        """
        Train LightGBM supervised fraud detection model.
        
        Args:
            features: Training features
            labels: Fraud labels (0=legitimate, 1=fraud)
            
        Returns:
            Training metrics and model info
        """
        self.logger.info("Training supervised LightGBM model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # LightGBM parameters optimized for fraud detection
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 64,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 100,
            'min_child_weight': 0.01,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'class_weight': 'balanced',  # Handle imbalanced data
            'random_state': 42,
            'verbose': -1
        }
        
        # Create datasets
        train_dataset = lgb.Dataset(X_train, label=y_train)
        val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)
        
        # Train model with early stopping
        self.models['supervised'] = lgb.train(
            lgb_params,
            train_dataset,
            valid_sets=[train_dataset, val_dataset],
            valid_names=['train', 'val'],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100)
            ]
        )
        
        # Evaluate model
        train_pred = self.models['supervised'].predict(X_train)
        val_pred = self.models['supervised'].predict(X_val)
        
        metrics = {
            'train_auc': roc_auc_score(y_train, train_pred),
            'val_auc': roc_auc_score(y_val, val_pred),
            'val_precision': precision_score(y_val, (val_pred > 0.5).astype(int)),
            'val_recall': recall_score(y_val, (val_pred > 0.5).astype(int)),
            'val_f1': f1_score(y_val, (val_pred > 0.5).astype(int))
        }
        
        self.training_metrics['supervised'] = metrics
        
        self.logger.info(f"Supervised model training complete:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def train_unsupervised_model(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Train Isolation Forest unsupervised anomaly detection model.
        
        Args:
            features: Training features (legitimate transactions only)
            
        Returns:
            Training metrics and model info
        """
        self.logger.info("Training unsupervised Isolation Forest model...")
        
        # Use scaled features for Isolation Forest
        features_scaled = self.preprocess_features(features, fit_scaler=False)
        
        # Isolation Forest parameters
        iso_params = {
            'n_estimators': 200,
            'contamination': 0.01,  # Expected fraud rate
            'max_samples': 'auto',
            'max_features': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train model
        self.models['unsupervised'] = IsolationForest(**iso_params)
        self.models['unsupervised'].fit(features_scaled)
        
        # Evaluate on training data
        anomaly_scores = self.models['unsupervised'].decision_function(features_scaled)
        anomaly_labels = self.models['unsupervised'].predict(features_scaled)
        
        outlier_fraction = (anomaly_labels == -1).mean()
        
        metrics = {
            'outlier_fraction': outlier_fraction,
            'score_mean': anomaly_scores.mean(),
            'score_std': anomaly_scores.std(),
            'score_min': anomaly_scores.min(),
            'score_max': anomaly_scores.max()
        }
        
        self.training_metrics['unsupervised'] = metrics
        
        self.logger.info(f"Unsupervised model training complete:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def create_ensemble_model(self, supervised_weight: float = 0.7) -> Dict[str, Any]:
        """
        Create ensemble model combining supervised and unsupervised predictions.
        
        Args:
            supervised_weight: Weight for supervised model (0-1)
            
        Returns:
            Ensemble configuration
        """
        ensemble_config = {
            'supervised_weight': supervised_weight,
            'unsupervised_weight': 1.0 - supervised_weight,
            'decision_thresholds': {
                'low_risk': 0.2,
                'high_risk': 0.8
            }
        }
        
        self.models['ensemble_config'] = ensemble_config
        
        self.logger.info(f"Ensemble model created with weights: "
                        f"supervised={supervised_weight:.2f}, unsupervised={1-supervised_weight:.2f}")
        
        return ensemble_config
    
    def generate_shap_explanations(self, features: pd.DataFrame, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Generate SHAP explanations for model interpretability.
        
        Args:
            features: Training features
            sample_size: Number of samples for SHAP analysis
            
        Returns:
            SHAP explanation objects
        """
        self.logger.info("Generating SHAP explanations...")
        
        # Sample data for SHAP (computationally expensive)
        sample_features = features.sample(min(sample_size, len(features)), random_state=42)
        
        # Create SHAP explainer for LightGBM
        explainer = shap.TreeExplainer(self.models['supervised'])
        shap_values = explainer.shap_values(sample_features)
        
        # Store explainer and sample for later use
        self.models['shap_explainer'] = explainer
        self.models['shap_sample'] = sample_features
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        feature_importance_dict = dict(zip(self.feature_names, feature_importance))
        
        self.logger.info("Top 10 most important features:")
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:10]:
            self.logger.info(f"  {feature}: {importance:.4f}")
        
        return {
            'explainer': explainer,
            'sample_shap_values': shap_values,
            'feature_importance': feature_importance_dict
        }
    
    def export_models_to_onnx(self, export_path: str):
        """
        Export trained models to ONNX format for production serving.
        
        Args:
            export_path: Directory to save ONNX models
        """
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Exporting models to ONNX format at {export_path}")
        
        # Export LightGBM to ONNX
        try:
            # Create a dummy input for ONNX conversion
            dummy_input = np.random.random((1, len(self.feature_names))).astype(np.float32)
            
            # Convert LightGBM to ONNX
            onnx_model = convert_lightgbm(
                self.models['supervised'],
                initial_types=[('features', dummy_input.shape)],
                target_opset=11
            )
            
            # Save ONNX model
            onnx.save_model(onnx_model, str(export_path / 'supervised_model.onnx'))
            self.logger.info("Supervised model exported to ONNX")
            
        except Exception as e:
            self.logger.error(f"Error exporting supervised model to ONNX: {e}")
        
        # Save Isolation Forest as pickle (ONNX conversion not straightforward)
        joblib.dump(
            self.models['unsupervised'], 
            export_path / 'unsupervised_model.pkl'
        )
        
        # Save scaler
        joblib.dump(
            self.scalers['standard'],
            export_path / 'feature_scaler.pkl'
        )
        
        # Save ensemble configuration and metadata
        model_metadata = {
            'feature_names': self.feature_names,
            'ensemble_config': self.models['ensemble_config'],
            'training_metrics': self.training_metrics,
            'model_version': datetime.now().isoformat(),
            'framework_versions': {
                'lightgbm': lgb.__version__,
                'sklearn': '1.2.2'  # From requirements
            }
        }
        
        with open(export_path / 'model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        self.logger.info(f"All models and metadata exported to {export_path}")
    
    def cross_validate_models(self, features: pd.DataFrame, labels: pd.Series, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation to assess model performance.
        
        Args:
            features: Training features
            labels: Fraud labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        self.logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
            X_train_fold = features.iloc[train_idx]
            X_val_fold = features.iloc[val_idx]
            y_train_fold = labels.iloc[train_idx]
            y_val_fold = labels.iloc[val_idx]
            
            # Train fold model
            lgb_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 64,
                'learning_rate': 0.05,
                'random_state': 42,
                'verbose': -1
            }
            
            train_dataset = lgb.Dataset(X_train_fold, label=y_train_fold)
            val_dataset = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_dataset)
            
            fold_model = lgb.train(
                lgb_params,
                train_dataset,
                valid_sets=[val_dataset],
                num_boost_round=500,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Evaluate fold
            val_pred = fold_model.predict(X_val_fold)
            fold_auc = roc_auc_score(y_val_fold, val_pred)
            cv_scores.append(fold_auc)
            
            self.logger.info(f"Fold {fold + 1} AUC: {fold_auc:.4f}")
        
        cv_results = {
            'cv_scores': cv_scores,
            'mean_auc': np.mean(cv_scores),
            'std_auc': np.std(cv_scores),
            'cv_folds': cv_folds
        }
        
        self.logger.info(f"Cross-validation complete: "
                        f"Mean AUC = {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
        
        return cv_results


def main():
    """Main training pipeline."""
    # Configuration
    config = {
        'data_path': 'data/training_data.csv',
        'model_export_path': 'models/exported',
        'supervised_weight': 0.7
    }
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize trainer
    trainer = FraudDetectionModelTrainer(config)
    
    # Load training data
    features, labels = trainer.load_training_data(config['data_path'])
    
    # Train supervised model
    if labels is not None:
        trainer.train_supervised_model(features, labels)
        
        # Cross-validate
        cv_results = trainer.cross_validate_models(features, labels)
        
        # Generate SHAP explanations
        trainer.generate_shap_explanations(features)
    
    # Train unsupervised model (use only legitimate transactions)
    if labels is not None:
        legitimate_features = features[labels == 0]
    else:
        legitimate_features = features
    
    trainer.train_unsupervised_model(legitimate_features)
    
    # Create ensemble
    trainer.create_ensemble_model(config['supervised_weight'])
    
    # Export models
    trainer.export_models_to_onnx(config['model_export_path'])
    
    print("Model training complete!")


if __name__ == "__main__":
    main()
