# Data Pipeline - Synapse AI Fraud Detection

## Overview
The data pipeline is responsible for ingesting transaction events and computing real-time features for fraud detection.

## Components

### 1. Data Ingestion (`ingestion/`)
- Kafka consumer for transaction events
- JSON schema validation
- Event preprocessing and enrichment

### 2. Feature Engineering (`feature-engineering/`)
- Real-time stream processing with Apache Flink
- Velocity calculations (transactions per time window)
- Behavioral deviation analysis
- Geospatial distance calculations
- Device and network anomaly detection

## Event Schema

```json
{
  "transaction_id": "string",
  "timestamp": "ISO 8601 datetime",
  "amount": "float",
  "currency": "string (ISO 4217)",
  "merchant_id": "string",
  "user_id": "string",
  "billing_address": {
    "street": "string",
    "city": "string",
    "state": "string",
    "country": "string",
    "postal_code": "string"
  },
  "shipping_address": {
    "street": "string",
    "city": "string", 
    "state": "string",
    "country": "string",
    "postal_code": "string"
  },
  "ip_address": "string",
  "user_agent": "string",
  "device_fingerprint": "string"
}
```

## Features Computed

### Velocity Features
- `velocity_1m`: Transactions in last 1 minute
- `velocity_5m`: Transactions in last 5 minutes
- `velocity_1h`: Transactions in last 1 hour
- `velocity_24h`: Transactions in last 24 hours
- `velocity_7d`: Transactions in last 7 days

### Behavioral Features
- `amount_zscore`: Z-score of current amount vs user history
- `amount_mean_7d`: User's average transaction amount (7 days)
- `amount_std_7d`: Standard deviation of user's transactions (7 days)

### Geospatial Features
- `distance_last_transaction`: Distance from last transaction location
- `distance_avg_5_transactions`: Average distance from last 5 transactions
- `new_country`: Boolean flag for new country
- `new_city`: Boolean flag for new city

### Device & Network Features
- `unique_devices_24h`: Count of unique devices used in 24 hours
- `is_new_ip`: Boolean flag for new IP address
- `is_new_user_agent`: Boolean flag for new user agent
- `is_new_device`: Boolean flag for new device fingerprint

### Merchant Features
- `merchant_transaction_count`: Times user has transacted with merchant
- `merchant_first_transaction`: Boolean flag for first transaction with merchant

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure Kafka connection in `config.yaml`

3. Start the feature engineering service:
```bash
python feature-engineering/stream_processor.py
```

## Testing

Run unit tests:
```bash
pytest tests/
```
