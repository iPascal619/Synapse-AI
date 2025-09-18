# Synapse AI - Fraud Detection Platform

##  World-Class Real-Time AI-Powered Fraud Detection

Synapse AI is an enterprise-grade, real-time fraud detection platform designed for e-commerce and online payment systems. Built with cutting-edge ML algorithms and designed for sub-100ms response times.

###  Key Features

- **Real-Time Detection**: Sub-100ms fraud scoring with explainable AI
- **Hybrid Intelligence**: ML models + business rules for comprehensive coverage
- **Continuous Learning**: Automated retraining with feedback loops
- **Enterprise Scale**: Handles 10,000+ transactions per second
- **Full Observability**: Comprehensive monitoring and alerting

### Performance Metrics

- **AUC**: >0.95 (Area Under ROC Curve)
- **Recall**: >90% (Fraud Detection Rate)
- **False Positive Rate**: <0.5%
- **API Latency**: 99th percentile <100ms
- **Throughput**: 10,000+ TPS


### Components

1. **Data Pipeline**: Kafka ingestion + real-time feature engineering
2. **ML Models**: LightGBM + Isolation Forest ensemble with ONNX serving
3. **Decision Engine**: FastAPI service with business rules integration
4. **Investigation Dashboard**: React/Next.js interface for fraud analysts
5. **Feedback Loop**: Continuous learning with MLflow and A/B testing
6. **Monitoring**: Prometheus + Grafana + AlertManager

## Key Features

- **Real-Time Performance**: Decisions made in under 100 milliseconds
- **Hybrid Intelligence**: Combines ML models, business rules, and human feedback
- **Explainable AI**: All high-risk decisions justified for compliance and review
- **Continuous Learning**: Automatically improves performance over time
- **Self-Learning**: Adapts to zero-day attacks and new fraud patterns

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Data Ingestion  │───▶│ Feature Engine   │───▶│ ML Ensemble     │
│ (Kafka/Kinesis) │    │ (Flink/Spark)    │    │ (LightGBM +     │
└─────────────────┘    └──────────────────┘    │ Isolation Forest│
                                               └─────────────────┘
                                                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Investigation   │◀───│ Decision Engine  │◀───│ Real-Time API   │
│ Dashboard       │    │ & Rules Engine   │    │ (<100ms)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │
        └───────────────────────┼─────────────────────────┐
                                │                         │
                        ┌──────────────────┐    ┌─────────────────┐
                        │ Feedback Loop    │    │ Retraining      │
                        │ & Human Labels   │    │ Pipeline        │
                        └──────────────────┘    └─────────────────┘
```

## Core Components

### 1. Data Ingestion Pipeline
- High-throughput stream processing (Apache Kafka/AWS Kinesis)
- Handles tens of thousands of events per second
- JSON schema validation

### 2. Real-Time Feature Engineering
- Stream processing with Apache Flink/Spark Streaming
- Computes velocity, behavioral deviation, geospatial, and device features
- Stateful processing for user behavior patterns

### 3. ML Model Ensemble
- **Supervised Model**: LightGBM for fraud classification
- **Unsupervised Model**: Isolation Forest for anomaly detection
- Model serving with ONNX/TensorFlow Serving

### 4. Decision Engine & API
- FastAPI service with <100ms response time
- Flexible rules engine for business logic
- SHAP-powered explainable AI

##  Dashboard Features

### Synapse AI Investigation Dashboard

The professional fraud analyst dashboard provides:

- **Real-time fraud analytics** with interactive charts
- **Transaction monitoring** with detailed fraud scoring
- **Beautiful responsive design** with Tailwind CSS
- **Optimized Synapse AI branding** with clean logo integration
- **Dark/light theme support**
- **Export capabilities** for reports and data analysis

### Dashboard Screenshots

The dashboard includes:
- **Fraud Detection Statistics** - Real-time metrics and KPIs
- **Transaction Trends** - Interactive charts showing fraud patterns
- **Recent Transactions** - Detailed transaction reviews with risk scores
- **Alert Management** - Priority-based fraud alert system

### Dashboard Setup (Detailed)

```bash
# Navigate to dashboard directory
cd dashboard

# Install dependencies (recreates node_modules)
npm install

# Available scripts:
npm run dev          # Start development server (http://localhost:3000)
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
npm run type-check   # TypeScript type checking
```

**Dashboard Environment Variables:**
```env
# dashboard/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=Synapse AI Fraud Detection
NEXT_PUBLIC_VERSION=1.0.0
```

**Production Deployment:**
```bash
# Build optimized version
npm run build

# Start production server
npm start

# Or deploy to Vercel/Netlify
npm run build && npm run export
```

### 6. Feedback Loop & Retraining
- Automated model retraining pipeline
- A/B testing for model deployment
- Continuous learning from analyst feedback

## Performance Targets

- **AUC Score**: > 0.95
- **Recall**: > 90% (fraud detection rate)
- **False Positive Rate**: < 0.5%
- **API Latency**: 99th percentile < 100ms
- **MTTR**: 50% reduction in fraud case resolution time

## Technology Stack

- **Stream Processing**: Apache Kafka, Apache Flink
- **ML Framework**: LightGBM, Scikit-learn, ONNX
- **API Service**: FastAPI, Pydantic
- **Frontend**: React, Next.js, TypeScript
- **Database**: PostgreSQL, Redis
- **Deployment**: Docker, Kubernetes
- **Monitoring**: Prometheus, Grafana

## 🚀 Quick Start & Setup

### Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 16 or higher
- **Docker**: Latest version (for containerized deployment)
- **Git**: For cloning the repository

### 1. Clone the Repository

```bash
git clone https://github.com/iPascal619/Synapse-AI.git
cd Synapse-AI/fraud-detection-ai
```

### 2. Environment Setup

#### Option A: Local Development Setup

**Backend Setup:**
```bash
# Navigate to API service
cd api-service

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your configuration
```

**Dashboard Setup:**
```bash
# Navigate to dashboard
cd dashboard

# Install Node.js dependencies
npm install

# Start development server
npm run dev
```

**Data Pipeline Setup:**
```bash
# Navigate to data pipeline
cd data-pipeline

# Install dependencies
pip install -r requirements.txt

# Configure Kafka (if using local setup)
# See data-pipeline/README.md for detailed instructions
```

#### Option B: Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 3. Configuration

#### API Service Configuration
Edit `api-service/.env`:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/fraud_db
REDIS_URL=redis://localhost:6379
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
MODEL_PATH=./models/
API_KEY=your_api_key_here
```

#### Dashboard Configuration
Edit `dashboard/.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=Synapse AI Fraud Detection
```

### 4. Database Setup

```bash
# Start PostgreSQL (if using Docker)
docker run -d --name postgres \
  -e POSTGRES_USER=fraud_user \
  -e POSTGRES_PASSWORD=fraud_pass \
  -e POSTGRES_DB=fraud_db \
  -p 5432:5432 postgres:13

# Run database migrations
cd api-service
python -m alembic upgrade head
```

### 5. Start Services

#### Development Mode:

**Terminal 1 - API Service:**
```bash
cd api-service
python main.py
# API available at http://localhost:8000
```

**Terminal 2 - Dashboard:**
```bash
cd dashboard
npm run dev
# Dashboard available at http://localhost:3000
```

**Terminal 3 - Data Pipeline:**
```bash
cd data-pipeline
python ingestion/kafka_ingestion.py
```

#### Production Mode:
```bash
# Start all services with Docker Compose
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 6. Verify Installation

#### Test API Endpoint:
```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key" \
  -d '{
    "transaction_id": "test_123",
    "amount": 1000.00,
    "merchant_id": "merchant_456",
    "user_id": "user_789"
  }'
```

#### Access Dashboard:
1. Open browser to `http://localhost:3000`
2. Login with demo credentials (see dashboard/README.md)
3. View fraud detection analytics

### 7. Load Sample Data

```bash
# Generate and load test transactions
cd scripts
python data_acquisition.py --sample-size 10000

# Run model training on sample data
python train_multi_dataset.py
```

## 📁 Detailed Component Setup

### Data Pipeline
```bash
cd data-pipeline
pip install -r requirements.txt

# Start Kafka (using Docker)
docker run -d --name kafka \
  -p 9092:9092 \
  -e KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 \
  confluentinc/cp-kafka:latest

# Start feature engineering
python feature-engineering/stream_processor.py
```

### ML Models
```bash
cd ml-models
pip install -r requirements.txt

# Train initial models
python training/train_models.py

# Start model serving
python serving/model_inference.py
```

### Monitoring Setup
```bash
# Start monitoring stack
cd infrastructure
docker-compose -f monitoring/docker-compose.monitoring.yml up -d

# Access Grafana: http://localhost:3001 (admin/admin)
# Access Prometheus: http://localhost:9090
```

## 🔧 Troubleshooting

### Common Issues:

**Port Already in Use:**
```bash
# Kill process on port 3000 (Dashboard)
npx kill-port 3000

# Kill process on port 8000 (API)
npx kill-port 8000
```

**Database Connection Issues:**
- Verify PostgreSQL is running: `docker ps | grep postgres`
- Check connection string in `.env` file
- Ensure database exists: `createdb fraud_db`

**Node Modules Issues:**
```bash
# Clear npm cache and reinstall
cd dashboard
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

**Python Dependencies:**
```bash
# Upgrade pip and reinstall
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Getting Help

- 📖 Check component-specific READMEs in each directory
- 🐛 Report issues on GitHub: [Issues](https://github.com/iPascal619/Synapse-AI/issues)
- 📧 Contact: [your-email@domain.com]

## 🎯 Next Steps

After setup:
1. **Configure data sources** - Connect your transaction data
2. **Train models** - Use your historical data for better accuracy  
3. **Customize rules** - Adapt business logic to your use case
4. **Scale deployment** - Use Kubernetes for production workloads
5. **Monitor performance** - Set up alerts and dashboards

## Directory Structure

```
fraud-detection-ai/
├── data-pipeline/          # Stream processing and feature engineering
├── ml-models/             # Machine learning models and training
├── api-service/           # FastAPI decision engine
├── dashboard/             # React investigation dashboard  
├── infrastructure/        # Docker, Kubernetes, monitoring
├── docs/                  # Architecture and API documentation
├── tests/                 # Unit and integration tests
└── scripts/              # Deployment and utility scripts
```

## License

MIT License - see LICENSE file for details.
