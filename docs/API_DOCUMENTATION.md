# API Documentation

## Overview

FraudGuard AI provides a comprehensive RESTful API for fraud detection and system management.

## Base URL

`
http://localhost:8000/api/v1
`

## Authentication

All API requests require authentication using API keys.

### API Key Header

`http
Authorization: Bearer YOUR_API_KEY
`

### Obtaining an API Key

[To be added: API key generation procedures]

## Core Endpoints

### Fraud Detection

#### Predict Fraud

`http
POST /api/v1/predict
`

Analyze a transaction for potential fraud.

**Request Body**:
`json
{
  "transaction_id": "txn_123456",
  "amount": 150.00,
  "merchant": "Online Store",
  "merchant_category": "retail",
  "timestamp": "2024-01-15T10:30:00Z",
  "user_id": "user_789",
  "card_number_hash": "hash_value",
  "location": {
    "country": "US",
    "city": "New York",
    "ip_address": "192.168.1.1"
  }
}
`

**Response**:
`json
{
  "transaction_id": "txn_123456",
  "fraud_score": 0.85,
  "is_fraud": true,
  "risk_level": "high",
  "confidence": 0.92,
  "factors": [
    "unusual_amount",
    "new_merchant",
    "velocity_check_failed"
  ],
  "recommended_action": "block",
  "timestamp": "2024-01-15T10:30:01Z"
}
`

**Status Codes**:
- 200 OK: Successful prediction
- 400 Bad Request: Invalid request data
- 401 Unauthorized: Missing or invalid API key
- 429 Too Many Requests: Rate limit exceeded
- 500 Internal Server Error: Server error

### Health Checks

#### Liveness Check

`http
GET /api/v1/health/live
`

Check if the application is running.

**Response**:
`json
{
  "status": "alive",
  "timestamp": "2024-01-15T10:30:00Z"
}
`

#### Readiness Check

`http
GET /api/v1/health/ready
`

Check if the application is ready to handle requests.

**Response**:
`json
{
  "status": "ready",
  "checks": {
    "database": "ok",
    "cache": "ok",
    "model": "ok"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
`

### Metrics

#### Get System Metrics

`http
GET /api/v1/metrics
`

Retrieve system performance metrics.

**Response**:
`json
{
  "requests_total": 10000,
  "requests_per_second": 50,
  "average_latency_ms": 45,
  "error_rate": 0.01,
  "model_inference_time_ms": 12,
  "cache_hit_rate": 0.85,
  "timestamp": "2024-01-15T10:30:00Z"
}
`

### Transaction Management

#### Get Transaction

`http
GET /api/v1/transactions/{transaction_id}
`

Retrieve transaction details.

**Response**:
`json
{
  "transaction_id": "txn_123456",
  "amount": 150.00,
  "merchant": "Online Store",
  "fraud_score": 0.85,
  "status": "blocked",
  "timestamp": "2024-01-15T10:30:00Z"
}
`

#### List Transactions

`http
GET /api/v1/transactions?limit=100&offset=0&status=flagged
`

List transactions with optional filtering.

**Query Parameters**:
- limit: Number of results (default: 100, max: 1000)
- offset: Pagination offset (default: 0)
- status: Filter by status (all, flagged, blocked, approved)
- start_date: Filter by start date (ISO 8601)
- end_date: Filter by end date (ISO 8601)

### Model Management

#### Get Model Info

`http
GET /api/v1/models/{model_id}
`

Retrieve model information and performance metrics.

**Response**:
`json
{
  "model_id": "xgboost_v1",
  "version": "1.0.0",
  "accuracy": 0.95,
  "precision": 0.93,
  "recall": 0.91,
  "f1_score": 0.92,
  "last_trained": "2024-01-10T00:00:00Z",
  "status": "active"
}
`

## Rate Limiting

API requests are rate-limited to ensure fair usage and system stability.

**Limits**:
- Standard tier: 1000 requests per minute
- Premium tier: 10000 requests per minute

**Rate Limit Headers**:
`http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1642248600
`

## Error Handling

All errors follow a consistent format:

`json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Missing required field: transaction_id",
    "details": {
      "field": "transaction_id",
      "reason": "required"
    }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
`

**Common Error Codes**:
- INVALID_REQUEST: Malformed request
- UNAUTHORIZED: Authentication failed
- FORBIDDEN: Insufficient permissions
- NOT_FOUND: Resource not found
- RATE_LIMIT_EXCEEDED: Too many requests
- INTERNAL_ERROR: Server error

## Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## SDKs and Client Libraries

[To be added: Available SDKs and client libraries]

## Webhooks

[To be added: Webhook configuration and usage]

## Best Practices

[To be added: API usage best practices]

## Changelog

[To be added: API version history and changes]
