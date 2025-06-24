# API Quick Start Guide

## 1. Get Your API Key

### Register a new user:
```bash
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_username",
    "email": "your@email.com",
    "password": "your_password"
  }'
```

**Response:**
```json
{
  "user_id": 1,
  "username": "your_username", 
  "api_key": "llm_abc123def456...",
  "message": "User registered successfully"
}
```

**Save your API key** - you'll need it for authenticated requests.

## 2. Use Your API Key

Include your API key in the `X-API-Key` header for authenticated endpoints:

```bash
X-API-Key: llm_abc123def456...
```

## 3. Generate Text

```bash
curl -X POST http://localhost:5000/api/models/1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "prompt": "def fibonacci(n):",
    "temperature": 0.7,
    "max_length": 100
  }'
```

## 4. Create Coding Training Job

```bash
curl -X POST http://localhost:5000/api/training/coding \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "model_id": 1,
    "job_name": "Python Code Training",
    "dataset_id": 1,
    "epochs": 5,
    "learning_rate": 0.00005,
    "batch_size": 4
  }'
```

## 5. Python Example

```python
import requests

API_KEY = "your_api_key_here"
BASE_URL = "http://localhost:5000/api"

headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

# Generate text
response = requests.post(
    f"{BASE_URL}/models/1/generate",
    json={
        "prompt": "def calculate_sum(a, b):",
        "temperature": 0.3,
        "max_length": 150
    },
    headers=headers
)

if response.status_code == 200:
    result = response.json()
    print(f"Generated: {result['text']}")
else:
    print(f"Error: {response.json()}")
```

## Rate Limiting

- **Default**: 1000 requests per day per API key
- **Resets**: Daily at midnight UTC
- **Custom limits**: Set when creating API keys
- **Monitor usage**: `GET /api/auth/keys`

## Available Endpoints

### No Authentication Required:
- `GET /api/health` - Health check
- `GET /api/models` - List models
- `GET /api/datasets` - List coding datasets
- `GET /api/training` - List training jobs
- `GET /api/statistics` - Platform stats

### Authentication Required:
- `POST /api/models/{id}/generate` - Generate text
- `POST /api/training` - Create training job
- `POST /api/training/coding` - Create coding training
- `POST /api/models/{id}/evaluate` - Run evaluation
- `GET /api/auth/keys` - List your API keys
- `POST /api/auth/keys` - Create new API key

## Error Responses

```json
{
  "error": "API key required"
}
```

Common status codes:
- `401` - Missing/invalid API key
- `400` - Bad request (missing fields)
- `429` - Rate limit exceeded
- `404` - Resource not found
- `500` - Server error

## Testing Your Setup

1. Register and get your API key
2. Test with a simple generation request
3. Monitor usage via `/api/auth/keys`
4. Create training jobs for your models

Visit `/api-docs` in your browser for the complete API documentation with examples.