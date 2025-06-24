# Complete API Usage Guide - Authentication & Rate Limiting

## Overview

The LLM Platform provides authenticated REST API endpoints with rate limiting for programmatic access to models, training, and evaluation features.

## Authentication System

### 1. User Registration

Register once to get your permanent API key:

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

### 2. Using Your API Key

Include your API key in the `X-API-Key` header:

```bash
X-API-Key: llm_abc123def456...
Content-Type: application/json
```

## Rate Limiting

- **Default**: 1000 requests per day per API key
- **Reset**: Daily at midnight UTC
- **Monitoring**: Check usage via `/api/auth/keys`
- **Custom Limits**: Set when creating additional API keys

## Core API Endpoints

### Text Generation (Authenticated)

Generate text using your models:

```bash
curl -X POST http://localhost:5000/api/models/1/generate \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "def fibonacci(n):",
    "temperature": 0.7,
    "max_length": 150,
    "top_p": 0.9,
    "top_k": 50
  }'
```

**Response:**
```json
{
  "text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "model_id": 1,
  "model_name": "GPT-2 Base",
  "generation_time": 1.23,
  "timestamp": "2025-06-24T14:24:00.000000"
}
```

### Coding Training Jobs (Authenticated)

Create specialized training for code generation:

**List available datasets:**
```bash
curl -X GET http://localhost:5000/api/datasets
```

**Create coding training:**
```bash
curl -X POST http://localhost:5000/api/training/coding \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": 1,
    "job_name": "Python Code Training",
    "dataset_id": 1,
    "epochs": 5,
    "learning_rate": 0.00005,
    "batch_size": 4,
    "lora_r": 16,
    "lora_alpha": 32
  }'
```

### API Key Management

**List your keys:**
```bash
curl -X GET http://localhost:5000/api/auth/keys \
  -H "X-API-Key: your_api_key"
```

**Create additional keys:**
```bash
curl -X POST http://localhost:5000/api/auth/keys \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "key_name": "Production Key",
    "rate_limit": 5000
  }'
```

## Python Integration

```python
import requests

class LLMPlatformAPI:
    def __init__(self, api_key, base_url="http://localhost:5000/api"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }
    
    def generate_code(self, prompt, **kwargs):
        """Generate code completion"""
        data = {"prompt": prompt, **kwargs}
        response = requests.post(
            f"{self.base_url}/models/1/generate",
            json=data,
            headers=self.headers
        )
        return response.json()
    
    def train_coding_model(self, model_id, dataset_id, job_name, **params):
        """Start coding-specific training"""
        data = {
            "model_id": model_id,
            "job_name": job_name,
            "dataset_id": dataset_id,
            **params
        }
        response = requests.post(
            f"{self.base_url}/training/coding",
            json=data,
            headers=self.headers
        )
        return response.json()
    
    def monitor_usage(self):
        """Check API usage"""
        response = requests.get(
            f"{self.base_url}/auth/keys",
            headers=self.headers
        )
        return response.json()

# Example usage
api = LLMPlatformAPI("your_api_key_here")

# Generate code
result = api.generate_code(
    prompt="class BinarySearchTree:",
    temperature=0.3,
    max_length=200
)

# Start training
training = api.train_coding_model(
    model_id=1,
    dataset_id=1,
    job_name="Advanced Python Training",
    epochs=5
)

# Monitor usage
usage = api.monitor_usage()
```

## Available Endpoints

### Public (No Authentication)
- `GET /api/health` - Health check
- `GET /api/models` - List models
- `GET /api/datasets` - List coding datasets
- `GET /api/training` - List training jobs
- `GET /api/statistics` - Platform stats

### Authenticated (Requires API Key)
- `POST /api/models/{id}/generate` - Generate text
- `POST /api/training` - Create training job
- `POST /api/training/coding` - Create coding training
- `POST /api/models/{id}/evaluate` - Run evaluation
- `GET /api/auth/keys` - List API keys
- `POST /api/auth/keys` - Create API key
- `DELETE /api/auth/keys/{id}` - Deactivate key

## Error Handling

**Common responses:**

```json
// Missing API key
{
  "error": "API key required"
}

// Invalid API key
{
  "error": "Invalid API key"
}

// Rate limit exceeded
{
  "error": "Rate limit exceeded"
}

// Missing fields
{
  "error": "Missing required field: model_id"
}
```

**HTTP Status Codes:**
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `429` - Rate Limit Exceeded
- `500` - Server Error

## Getting Started

1. **Register**: Get your API key via `/api/auth/register`
2. **Test**: Try text generation with your key
3. **Train**: Create coding training jobs
4. **Monitor**: Check usage via `/api/auth/keys`
5. **Scale**: Create additional keys with custom limits

## Live Documentation

Visit `/api-docs` in your browser for interactive documentation with live examples and testing interface.

Your API is ready for production use with authentication, rate limiting, and comprehensive error handling.