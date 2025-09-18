import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import asyncio
import redis.asyncio as redis
import json
from scipy import stats


class ModelDriftDetector:
    """
    Detects data drift and model performance drift in the fraud detection system.
    
    Monitors feature distributions and model performance metrics to trigger
    retraining when significant drift is detected.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize drift detector."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Drift detection thresholds
        self.feature_drift_threshold = config.get('feature_drift_threshold', 0.05)
        self.performance_drift_threshold = config.get('performance_drift_threshold', 0.02)
        self.min_samples_for_detection = config.get('min_samples_for_detection', 1000)
        
        # Statistical test parameters
        self.alpha = config.get('statistical_alpha', 0.05)
        
    async def collect_recent_features(self, hours: int = 24) -> pd.DataFrame:
        """
        Collect recent feature data for drift analysis.
        
        Args:
            hours: Number of hours of recent data to collect
            
        Returns:
            DataFrame with recent feature vectors
        """
        self.logger.info(f"Collecting features from last {hours} hours")
        
        # In production, this would query the feature store
        # For demo, generate mock recent data
        feature_names = [
            'velocity_1m', 'velocity_5m', 'velocity_1h', 'velocity_24h', 'velocity_7d',
            'amount_zscore', 'amount_mean_7d', 'amount_std_7d',
            'distance_last_transaction', 'distance_avg_5_transactions',
            'unique_devices_24h', 'merchant_transaction_count'
        ]
        
        # Simulate some drift by shifting distributions
        drift_factor = 0.1 if hours <= 24 else 0.0
        
        recent_data = []
        for i in range(self.min_samples_for_detection):
            features = {}
            for feature in feature_names:
                if 'velocity' in feature:
                    # Simulate higher velocity (potential attack)
                    features[feature] = np.random.poisson(2 + drift_factor * 3)
                elif 'amount' in feature:
                    # Simulate amount distribution shift
                    features[feature] = np.random.normal(0 + drift_factor, 1)
                elif 'distance' in feature:
                    features[feature] = np.random.exponential(100 + drift_factor * 50)
                else:
                    features[feature] = np.random.random() + drift_factor * 0.2
            
            recent_data.append(features)
        
        recent_df = pd.DataFrame(recent_data)
        self.logger.info(f"Collected {len(recent_df)} recent feature vectors")
        
        return recent_df
    
    def load_baseline_features(self) -> pd.DataFrame:
        """
        Load baseline feature distributions from training data.
        
        Returns:
            DataFrame with baseline feature vectors
        """
        self.logger.info("Loading baseline feature distributions")
        
        # In production, load from feature store or training data
        # For demo, generate mock baseline data
        feature_names = [
            'velocity_1m', 'velocity_5m', 'velocity_1h', 'velocity_24h', 'velocity_7d',
            'amount_zscore', 'amount_mean_7d', 'amount_std_7d',
            'distance_last_transaction', 'distance_avg_5_transactions',
            'unique_devices_24h', 'merchant_transaction_count'
        ]
        
        baseline_data = []
        for i in range(self.min_samples_for_detection):
            features = {}
            for feature in feature_names:
                if 'velocity' in feature:
                    features[feature] = np.random.poisson(2)
                elif 'amount' in feature:
                    features[feature] = np.random.normal(0, 1)
                elif 'distance' in feature:
                    features[feature] = np.random.exponential(100)
                else:
                    features[feature] = np.random.random()
            
            baseline_data.append(features)
        
        baseline_df = pd.DataFrame(baseline_data)
        self.logger.info(f"Loaded {len(baseline_df)} baseline feature vectors")
        
        return baseline_df
    
    def detect_feature_drift(self, baseline_df: pd.DataFrame, recent_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift in individual features using statistical tests.
        
        Args:
            baseline_df: Baseline feature distributions
            recent_df: Recent feature distributions
            
        Returns:
            Drift detection results
        """
        self.logger.info("Detecting feature drift using statistical tests")
        
        drift_results = {
            'features_with_drift': [],
            'drift_scores': {},
            'p_values': {},
            'overall_drift_detected': False
        }
        
        for feature in baseline_df.columns:
            if feature in recent_df.columns:
                baseline_values = baseline_df[feature].values
                recent_values = recent_df[feature].values
                
                # Use Kolmogorov-Smirnov test for continuous features
                if len(np.unique(baseline_values)) > 10:  # Continuous feature
                    statistic, p_value = stats.ks_2samp(baseline_values, recent_values)
                else:  # Categorical or discrete feature
                    statistic, p_value = stats.chi2_contingency([
                        np.histogram(baseline_values, bins=10)[0],
                        np.histogram(recent_values, bins=10)[0]
                    ])[:2]
                
                drift_results['drift_scores'][feature] = float(statistic)
                drift_results['p_values'][feature] = float(p_value)
                
                # Check if drift is significant
                if p_value < self.alpha:
                    drift_results['features_with_drift'].append(feature)
                    self.logger.warning(f"Drift detected in feature {feature}: p-value={p_value:.4f}")
        
        # Overall drift if significant number of features show drift
        drift_ratio = len(drift_results['features_with_drift']) / len(baseline_df.columns)
        drift_results['overall_drift_detected'] = drift_ratio > self.feature_drift_threshold
        drift_results['drift_ratio'] = drift_ratio
        
        self.logger.info(f"Feature drift analysis complete: {len(drift_results['features_with_drift'])} features with drift")
        
        return drift_results
    
    async def collect_recent_predictions(self, hours: int = 24) -> pd.DataFrame:
        """
        Collect recent model predictions and outcomes.
        
        Args:
            hours: Number of hours of recent data
            
        Returns:
            DataFrame with predictions and actual outcomes
        """
        self.logger.info(f"Collecting recent predictions from last {hours} hours")
        
        # In production, query prediction logs and feedback
        # For demo, generate mock prediction data
        prediction_data = []
        
        for i in range(1000):
            # Simulate some performance drift
            base_accuracy = 0.95
            drift_factor = 0.02 if hours <= 24 else 0.0
            current_accuracy = base_accuracy - drift_factor
            
            prediction = np.random.random()
            actual = 1 if np.random.random() < 0.01 else 0  # 1% fraud rate
            
            # Simulate degraded performance
            if actual == 1 and np.random.random() > current_accuracy:
                prediction = np.random.random() * 0.5  # False negative
            elif actual == 0 and np.random.random() > current_accuracy:
                prediction = 0.5 + np.random.random() * 0.5  # False positive
            
            prediction_data.append({
                'transaction_id': f'tx_{i}',
                'prediction': prediction,
                'actual': actual,
                'timestamp': datetime.now() - timedelta(hours=np.random.uniform(0, hours))
            })
        
        predictions_df = pd.DataFrame(prediction_data)
        self.logger.info(f"Collected {len(predictions_df)} recent predictions")
        
        return predictions_df
    
    def detect_performance_drift(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift in model performance metrics.
        
        Args:
            predictions_df: Recent predictions and outcomes
            
        Returns:
            Performance drift detection results
        """
        self.logger.info("Detecting performance drift")
        
        # Calculate recent performance metrics
        y_true = predictions_df['actual'].values
        y_pred = predictions_df['prediction'].values
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        
        recent_metrics = {
            'auc': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5,
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true, y_pred_binary, zero_division=0)
        }
        
        # Load baseline metrics (from training or previous period)
        baseline_metrics = {
            'auc': 0.95,
            'precision': 0.89,
            'recall': 0.91,
            'f1': 0.90
        }
        
        # Calculate performance drift
        drift_results = {
            'performance_drift_detected': False,
            'recent_metrics': recent_metrics,
            'baseline_metrics': baseline_metrics,
            'metric_changes': {}
        }
        
        significant_drift = False
        for metric in baseline_metrics:
            change = baseline_metrics[metric] - recent_metrics[metric]
            drift_results['metric_changes'][metric] = change
            
            if change > self.performance_drift_threshold:
                significant_drift = True
                self.logger.warning(f"Performance drift in {metric}: {change:.4f} degradation")
        
        drift_results['performance_drift_detected'] = significant_drift
        
        self.logger.info(f"Performance drift analysis complete: drift_detected={significant_drift}")
        
        return drift_results
    
    async def run_drift_detection(self) -> Dict[str, Any]:
        """
        Run complete drift detection analysis.
        
        Returns:
            Comprehensive drift detection results
        """
        self.logger.info("Starting drift detection analysis")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'feature_drift': {},
            'performance_drift': {},
            'recommendation': 'no_action'
        }
        
        try:
            # Collect data
            recent_features = await self.collect_recent_features(24)
            baseline_features = self.load_baseline_features()
            recent_predictions = await self.collect_recent_predictions(24)
            
            # Detect feature drift
            feature_drift = self.detect_feature_drift(baseline_features, recent_features)
            results['feature_drift'] = feature_drift
            
            # Detect performance drift
            performance_drift = self.detect_performance_drift(recent_predictions)
            results['performance_drift'] = performance_drift
            
            # Determine recommendation
            if (feature_drift['overall_drift_detected'] or 
                performance_drift['performance_drift_detected']):
                results['recommendation'] = 'trigger_retraining'
                self.logger.warning("Significant drift detected - recommending model retraining")
            else:
                results['recommendation'] = 'continue_monitoring'
                self.logger.info("No significant drift detected - continue monitoring")
            
        except Exception as e:
            self.logger.error(f"Error in drift detection: {e}")
            results['error'] = str(e)
            results['recommendation'] = 'manual_investigation'
        
        return results
    
    async def store_drift_results(self, results: Dict[str, Any]):
        """Store drift detection results for monitoring dashboard."""
        try:
            # In production, store in database or monitoring system
            redis_client = redis.Redis(host='localhost', port=6379, db=0)
            
            await redis_client.setex(
                'drift_detection_latest',
                3600,  # 1 hour TTL
                json.dumps(results, default=str)
            )
            
            await redis_client.close()
            self.logger.info("Drift detection results stored")
            
        except Exception as e:
            self.logger.error(f"Error storing drift results: {e}")


async def main():
    """Main entry point for drift detection."""
    # Configuration
    config = {
        'feature_drift_threshold': 0.1,  # 10% of features must show drift
        'performance_drift_threshold': 0.02,  # 2% performance degradation
        'min_samples_for_detection': 1000,
        'statistical_alpha': 0.05
    }
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run drift detection
    detector = ModelDriftDetector(config)
    results = await detector.run_drift_detection()
    
    # Store results
    await detector.store_drift_results(results)
    
    print("Drift Detection Results:")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
