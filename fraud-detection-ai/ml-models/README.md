# ML Models - Synapse AI Fraud Detection

## Overview
The ML model ensemble combines supervised and unsupervised learning approaches for comprehensive fraud detection.

## Model Architecture

### 1. Supervised Learning Model
- **Algorithm**: LightGBM Gradient Boosting
- **Purpose**: Classify transactions as fraud/legitimate based on labeled training data
- **Input**: Rich feature vector from feature engineering pipeline
- **Output**: Fraud probability score (0.0 to 1.0)

### 2. Unsupervised Anomaly Detection
- **Algorithm**: Isolation Forest
- **Purpose**: Detect statistical outliers without requiring fraud labels
- **Input**: Feature vector from legitimate transactions only
- **Output**: Anomaly score and binary outlier flag

### 3. Model Ensemble
- **Combination**: Weighted average of supervised and unsupervised scores
- **Serving**: ONNX format for high-performance inference
- **Deployment**: Containerized with auto-scaling

## Performance Targets

- **AUC Score**: > 0.95
- **Recall**: > 90% (fraud detection rate)
- **Precision**: > 85% (avoid false positives)
- **Inference Latency**: < 10ms per prediction
- **Throughput**: > 10,000 predictions/second

## Training Pipeline

1. **Data Preparation**: Load and preprocess training data
2. **Feature Engineering**: Apply same transformations as real-time pipeline
3. **Model Training**: Train both supervised and unsupervised models
4. **Validation**: Cross-validation and holdout testing
5. **Model Export**: Convert to ONNX for serving
6. **A/B Testing**: Shadow deployment before production

## Model Serving

### ONNX Runtime
- High-performance inference engine
- Cross-platform compatibility
- Optimized for production workloads

### Model Versioning
- MLflow for experiment tracking
- Model registry for version management
- Blue-green deployment strategy

## Explainability

### SHAP (SHapley Additive exPlanations)
- Individual prediction explanations
- Feature importance for each transaction
- Global model interpretability

### Business Rules Integration
- Combine ML predictions with business logic
- Configurable rule thresholds
- Human-interpretable decision criteria

## Continuous Learning

### Feedback Loop
- Analyst labels feed back into training data
- Automated retraining pipeline
- Performance monitoring and alerting

### Model Drift Detection
- Statistical tests for feature drift
- Performance degradation alerts
- Automatic model retraining triggers
