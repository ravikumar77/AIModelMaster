# LLM Platform API Authentication Guide

## How to Use API Endpoints with Authentication and Rate Limiting

### Step 1: Register and Get Your API Key

**Register a new user:**
```bash
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_username",
    "email": "your@email.com", 
    "password": "your_password"
  }'
```

**Response includes your API key:**
```json
{
  "user_id": 5,
  "username": "your_username",
  "api_key": "llm_abc123def456ghi789...",
  "message": "User registered successfully"
}
```

**💡 Save your API key - you'll need it for all authenticated requests!**

### Step 2: Use API Key for Authentication

**Include your API key in request headers:**
```bash
X-API-Key: llm_abc123def456ghi789...
Content-Type: application/json
```

### Step 3: Generate Text (Authenticated)

```bash
curl -X POST http://localhost:5000/api/models/1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "prompt": "def calculate_fibonacci(n):",
    "temperature": 0.7,
    "max_length": 150,
    "top_p": 0.9,
    "top_k": 50
  }'
```

**Response:**
```json
{
  "text": "def calculate_fibonacci(n):\n    if n <= 1:\n        return n\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)",
  "model_id": 1,
  "model_name": "GPT-2 Base",
  "generation_time": 1.23,
  "parameters": {
    "temperature": 0.7,
    "max_length": 150
  },
  "timestamp": "2025-06-24T14:22:00.000000"
}
```

### Step 4: Create Coding Training Jobs (Authenticated)

**List available coding datasets first:**
```bash
curl -X GET http://localhost:5000/api/datasets
```

**Create a coding training job:**
```bash
curl -X POST http://localhost:5000/api/training/coding \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
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

### Step 5: Monitor Your API Usage

**Check your API keys and usage:**
```bash
curl -X GET http://localhost:5000/api/auth/keys \
  -H "X-API-Key: your_api_key_here"
```

**Response shows rate limit usage:**
```json
{
  "api_keys": [
    {
      "id": 1,
      "key_name": "Default Key",
      "usage_count": 15,
      "rate_limit": 1000,
      "last_used": "2025-06-24T14:22:00.000000"
    }
  ]
}
```

## Rate Limiting Details

- **Default Limit**: 1000 requests per day per API key
- **Reset Time**: Daily at midnight UTC
- **Custom Limits**: Set when creating new API keys
- **Rate Limit Exceeded**: Returns HTTP 401 with error message

**Create additional API keys with custom limits:**
```bash
curl -X POST http://localhost:5000/api/auth/keys \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "key_name": "High Volume Key",
    "rate_limit": 5000
  }'
```

## Python Code Example

```python
import requests
import json

class LLMPlatformAPI:
    def __init__(self, api_key, base_url="http://localhost:5000/api"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }
    
    def generate_text(self, model_id, prompt, **kwargs):
        """Generate text using a model"""
        data = {"prompt": prompt, **kwargs}
        response = requests.post(
            f"{self.base_url}/models/{model_id}/generate",
            json=data,
            headers=self.headers
        )
        return response.json()
    
    def create_coding_training(self, model_id, job_name, dataset_id, **kwargs):
        """Create a coding training job"""
        data = {
            "model_id": model_id,
            "job_name": job_name, 
            "dataset_id": dataset_id,
            **kwargs
        }
        response = requests.post(
            f"{self.base_url}/training/coding",
            json=data,
            headers=self.headers
        )
        return response.json()
    
    def check_usage(self):
        """Check API key usage"""
        response = requests.get(
            f"{self.base_url}/auth/keys",
            headers=self.headers
        )
        return response.json()

# Usage example
api = LLMPlatformAPI("your_api_key_here")

# Generate code
result = api.generate_text(
    model_id=1,
    prompt="def binary_search(arr, target):",
    temperature=0.3,
    max_length=200
)
print(f"Generated: {result['text']}")

# Create training job
job = api.create_coding_training(
    model_id=1,
    job_name="Python Advanced Training",
    dataset_id=1,
    epochs=5
)
print(f"Training job: {job['job_name']}")

# Check usage
usage = api.check_usage()
print(f"Usage: {usage['api_keys'][0]['usage_count']}/1000")
```

## Available Endpoints

### Public (No Authentication):
- `GET /api/health` - Health check
- `GET /api/models` - List models
- `GET /api/datasets` - List coding datasets  
- `GET /api/training` - List training jobs
- `GET /api/statistics` - Platform statistics

### Authenticated (Requires API Key):
- `POST /api/models/{id}/generate` - Generate text
- `POST /api/training` - Create training job
- `POST /api/training/coding` - Create coding training
- `POST /api/models/{id}/evaluate` - Run evaluation
- `GET /api/auth/keys` - List your API keys
- `POST /api/auth/keys` - Create new API key
- `DELETE /api/auth/keys/{id}` - Deactivate API key

## Error Handling

**Common error responses:**

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

// Missing required fields
{
  "error": "Missing required field: model_id"
}
```

**HTTP Status Codes:**
- `200` - Success
- `201` - Created  
- `400` - Bad Request
- `401` - Unauthorized (API key issue)
- `404` - Not Found
- `429` - Rate Limit Exceeded
- `500` - Server Error

## Testing Your Setup

1. **Register**: Get your API key
2. **Test Generation**: Try generating text
3. **Monitor Usage**: Check your rate limits
4. **Train Models**: Create coding training jobs

Visit the web interface at `/api-docs` for interactive documentation with more examples.