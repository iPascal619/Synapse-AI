# API Service - Synapse AI Fraud Detection

## Overview
The API service provides a real-time fraud detection endpoint with <100ms response time. It combines machine learning models with business rules and provides explainable AI features.

## API Endpoints

### POST /v1/score_transaction
Primary fraud detection endpoint that processes transactions in real-time.

**Request Body:**
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

**Response:**
```json
{
  "transaction_id": "string",
  "decision": "APPROVE" | "DENY" | "REVIEW",
  "risk_score": 0.85,
  "decision_details": {
    "model_score": 0.92,
    "rules_triggered": ["new_device_rule"],
    "explanation": {
      "main_contributing_features": {
        "velocity_5m_count": 5,
        "amount_zscore": 3.5,
        "is_new_device": true
      }
    }
  }
}
```

### GET /health
Service health check endpoint.

### GET /metrics
Prometheus metrics for monitoring.

### GET /models/info
Information about loaded models and versions.

## Performance Requirements

- **Latency**: 99th percentile < 100ms
- **Throughput**: > 10,000 requests/second
- **Availability**: 99.9% uptime
- **Accuracy**: AUC > 0.95

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Load Balancer   │───▶│ FastAPI Service  │───▶│ Feature Engine  │
│ (nginx/ALB)     │    │ (Multiple pods)  │    │ (Redis/Memory)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │
                       ┌──────────────────┐
                       │ ML Model Serving │
                       │ (ONNX Runtime)   │
                       └──────────────────┘
```

## Deployment

### Docker
```bash
docker build -t synapse-ai-api .
docker run -p 8000:8000 synapse-ai-api
```

### Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

## Monitoring

- **Metrics**: Prometheus + Grafana
- **Logging**: Structured JSON logs
- **Tracing**: OpenTelemetry (optional)
- **Alerting**: PagerDuty integration

## Security

- **API Keys**: Required for all endpoints
- **Rate Limiting**: Per-client request limits
- **Input Validation**: Pydantic schema validation
- **HTTPS**: TLS 1.3 encryption
- **Network**: VPC/private subnets only
