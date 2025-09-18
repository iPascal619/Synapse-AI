import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from datetime import datetime, timedelta
import joblib
import json
from pathlib import Path
import mlflow
import mlflow.lightgbm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from data_pipeline.schemas import FeedbackLabel
from ml_models.training.train_models import FraudDetectionModelTrainer


class AutomatedRetrainingPipeline:
    """
    Automated retraining pipeline for fraud detection models.
    
    Implements continuous learning by periodically retraining models
    with new labeled data from analyst feedback.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize retraining pipeline."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # MLflow configuration
        mlflow.set_tracking_uri(config.get('mlflow_uri', 'sqlite:///mlflow.db'))
        mlflow.set_experiment(config.get('experiment_name', 'fraud_detection_retraining'))
        
        self.min_new_samples = config.get('min_new_samples', 1000)
        self.performance_threshold = config.get('performance_threshold', 0.01)
        self.retraining_frequency_days = config.get('retraining_frequency_days', 7)
        
    def collect_feedback_data(self, since_date: datetime) -> pd.DataFrame:
        """
        Collect analyst feedback labels since the given date.
        
        Args:
            since_date: Collect feedback from this date onwards
            
        Returns:
            DataFrame with feedback labels
        """
        self.logger.info(f"Collecting feedback data since {since_date}")
        
        # In production, this would query the database
        # For demo, we'll simulate collecting feedback
        feedback_data = []
        
        # Mock feedback collection
        for i in range(self.min_new_samples):
            feedback_data.append({
                'transaction_id': f'tx_{i}',
                'is_fraud': np.random.choice([0, 1], p=[0.95, 0.05]),
                'analyst_id': f'analyst_{np.random.randint(1, 10)}',
                'timestamp': since_date + timedelta(hours=np.random.randint(1, 168)),
                'confidence': np.random.uniform(0.8, 1.0)
            })
        
        feedback_df = pd.DataFrame(feedback_data)
        self.logger.info(f"Collected {len(feedback_df)} feedback samples")
        
        return feedback_df
    
    def load_features_for_transactions(self, transaction_ids: List[str]) -> pd.DataFrame:
        """
        Load computed features for the given transaction IDs.
        
        Args:
            transaction_ids: List of transaction IDs
            
        Returns:
            DataFrame with features
        """
        self.logger.info(f"Loading features for {len(transaction_ids)} transactions")
        
        # In production, this would query the feature store
        # For demo, generate mock features
        feature_names = [
            'velocity_1m', 'velocity_5m', 'velocity_1h', 'velocity_24h', 'velocity_7d',
            'amount_zscore', 'amount_mean_7d', 'amount_std_7d',
            'distance_last_transaction', 'distance_avg_5_transactions',
            'new_country', 'new_city', 'unique_devices_24h',
            'is_new_ip', 'is_new_user_agent', 'is_new_device',
            'merchant_transaction_count', 'merchant_first_transaction'
        ]
        
        features_data = []
        for tx_id in transaction_ids:
            features = {
                'transaction_id': tx_id,
                **{name: np.random.random() for name in feature_names}
            }
            features_data.append(features)
        
        features_df = pd.DataFrame(features_data)
        self.logger.info(f"Loaded features: {features_df.shape}")
        
        return features_df
    
    def prepare_training_data(self, feedback_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data by combining feedback with features.
        
        Args:
            feedback_df: Analyst feedback labels
            
        Returns:
            Tuple of (features, labels)
        """
        # Load features for labeled transactions
        features_df = self.load_features_for_transactions(feedback_df['transaction_id'].tolist())
        
        # Merge feedback with features
        training_data = feedback_df.merge(features_df, on='transaction_id', how='inner')
        
        # Separate features and labels
        feature_columns = [col for col in training_data.columns 
                          if col not in ['transaction_id', 'is_fraud', 'analyst_id', 'timestamp', 'confidence']]
        
        features = training_data[feature_columns]
        labels = training_data['is_fraud']
        
        self.logger.info(f"Prepared training data: {features.shape[0]} samples, {features.shape[1]} features")
        return features, labels
    
    def load_current_model_performance(self) -> Dict[str, float]:
        """Load performance metrics of the current production model."""
        try:
            # In production, load from model registry
            current_metrics = {
                'auc': 0.952,
                'precision': 0.89,
                'recall': 0.91,
                'f1': 0.90
            }
            return current_metrics
        except Exception as e:
            self.logger.error(f"Error loading current model performance: {e}")
            return {}
    
    def train_candidate_model(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[Any, Dict[str, float]]:
        """
        Train a candidate model with the new data.
        
        Args:
            features: Training features
            labels: Training labels
            
        Returns:
            Tuple of (trained_model, performance_metrics)
        """
        self.logger.info("Training candidate model...")
        
        with mlflow.start_run(run_name=f"retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("training_samples", len(features))
            mlflow.log_param("fraud_rate", labels.mean())
            mlflow.log_param("retraining_date", datetime.now().isoformat())
            
            # Initialize trainer
            trainer_config = {
                'supervised_weight': 0.7
            }
            trainer = FraudDetectionModelTrainer(trainer_config)
            trainer.feature_names = features.columns.tolist()
            
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Train supervised model
            metrics = trainer.train_supervised_model(X_train, y_train)
            
            # Evaluate on validation set
            val_pred = trainer.models['supervised'].predict(X_val)
            
            val_metrics = {
                'val_auc': roc_auc_score(y_val, val_pred),
                'val_precision': precision_score(y_val, (val_pred > 0.5).astype(int)),
                'val_recall': recall_score(y_val, (val_pred > 0.5).astype(int)),
                'val_f1': f1_score(y_val, (val_pred > 0.5).astype(int))
            }
            
            # Log metrics
            for metric, value in val_metrics.items():
                mlflow.log_metric(metric, value)
            
            # Log model
            mlflow.lightgbm.log_model(
                trainer.models['supervised'], 
                "model",
                registered_model_name="fraud_detection_candidate"
            )
            
            self.logger.info(f"Candidate model trained with AUC: {val_metrics['val_auc']:.4f}")
            
            return trainer.models['supervised'], val_metrics
    
    def compare_model_performance(self, current_metrics: Dict[str, float], 
                                 candidate_metrics: Dict[str, float]) -> bool:
        """
        Compare candidate model performance against current model.
        
        Args:
            current_metrics: Current production model metrics
            candidate_metrics: Candidate model metrics
            
        Returns:
            True if candidate model should be deployed
        """
        if not current_metrics:
            self.logger.info("No current model metrics found, deploying candidate")
            return True
        
        # Check if candidate model significantly outperforms current model
        auc_improvement = candidate_metrics.get('val_auc', 0) - current_metrics.get('auc', 0)
        f1_improvement = candidate_metrics.get('val_f1', 0) - current_metrics.get('f1', 0)
        
        should_deploy = (
            auc_improvement > self.performance_threshold or
            f1_improvement > self.performance_threshold
        )
        
        self.logger.info(f"Performance comparison:")
        self.logger.info(f"  AUC improvement: {auc_improvement:.4f}")
        self.logger.info(f"  F1 improvement: {f1_improvement:.4f}")
        self.logger.info(f"  Should deploy: {should_deploy}")
        
        return should_deploy
    
    def deploy_model(self, model: Any, metrics: Dict[str, float]) -> bool:
        """
        Deploy the candidate model to production.
        
        Args:
            model: Trained model to deploy
            metrics: Model performance metrics
            
        Returns:
            True if deployment successful
        """
        try:
            self.logger.info("Deploying candidate model to production...")
            
            # Create model export directory
            export_path = Path(self.config.get('model_export_path', 'models/production'))
            export_path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = export_path / f'fraud_model_{timestamp}.pkl'
            joblib.dump(model, model_path)
            
            # Save metrics
            metrics_path = export_path / f'model_metrics_{timestamp}.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Update symlink to latest model
            latest_model_path = export_path / 'latest_model.pkl'
            if latest_model_path.exists():
                latest_model_path.unlink()
            latest_model_path.symlink_to(model_path.name)
            
            # Register model in MLflow
            client = mlflow.tracking.MlflowClient()
            model_version = client.transition_model_version_stage(
                name="fraud_detection_candidate",
                version="latest",
                stage="Production"
            )
            
            self.logger.info(f"Model deployed successfully: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deploying model: {e}")
            return False
    
    def run_retraining_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete retraining pipeline.
        
        Returns:
            Pipeline execution results
        """
        start_time = datetime.now()
        results = {
            'start_time': start_time.isoformat(),
            'status': 'failed',
            'deployed': False,
            'metrics': {}
        }
        
        try:
            self.logger.info("Starting automated retraining pipeline...")
            
            # Determine data collection window
            since_date = start_time - timedelta(days=self.retraining_frequency_days)
            
            # Collect new feedback data
            feedback_df = self.collect_feedback_data(since_date)
            
            if len(feedback_df) < self.min_new_samples:
                self.logger.info(f"Insufficient new samples ({len(feedback_df)} < {self.min_new_samples})")
                results['status'] = 'skipped_insufficient_data'
                return results
            
            # Prepare training data
            features, labels = self.prepare_training_data(feedback_df)
            
            # Load current model performance
            current_metrics = self.load_current_model_performance()
            
            # Train candidate model
            candidate_model, candidate_metrics = self.train_candidate_model(features, labels)
            results['metrics'] = candidate_metrics
            
            # Compare performance
            should_deploy = self.compare_model_performance(current_metrics, candidate_metrics)
            
            if should_deploy:
                # Deploy new model
                deployment_success = self.deploy_model(candidate_model, candidate_metrics)
                results['deployed'] = deployment_success
                
                if deployment_success:
                    results['status'] = 'success_deployed'
                    self.logger.info("Retraining pipeline completed successfully - model deployed")
                else:
                    results['status'] = 'success_not_deployed'
                    self.logger.info("Retraining pipeline completed but deployment failed")
            else:
                results['status'] = 'success_not_deployed'
                self.logger.info("Retraining pipeline completed - model not deployed (insufficient improvement)")
            
        except Exception as e:
            self.logger.error(f"Error in retraining pipeline: {e}")
            results['error'] = str(e)
        
        finally:
            results['end_time'] = datetime.now().isoformat()
            results['duration_minutes'] = (datetime.now() - start_time).total_seconds() / 60
        
        return results


def main():
    """Main entry point for retraining pipeline."""
    # Configuration
    config = {
        'mlflow_uri': 'sqlite:///fraud_detection_mlflow.db',
        'experiment_name': 'fraud_detection_retraining',
        'min_new_samples': 500,  # Minimum new labeled samples required
        'performance_threshold': 0.01,  # Minimum improvement required
        'retraining_frequency_days': 7,
        'model_export_path': 'models/production'
    }
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run retraining pipeline
    pipeline = AutomatedRetrainingPipeline(config)
    results = pipeline.run_retraining_pipeline()
    
    print("Retraining Pipeline Results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
