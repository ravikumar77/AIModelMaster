# LLM Platform API Documentation

## Base URL
```
http://localhost:5000/api
```

## Authentication
Currently, no authentication is required for API access.

## Error Responses
All error responses follow this format:
```json
{
  "error": "Error description"
}
```

## API Endpoints

### Health Check
**GET** `/api/health`

Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-24T13:45:30.123456",
  "version": "1.0.0"
}
```

### Models

#### List Models
**GET** `/api/models`

Get all available models.

**Response:**
```json
{
  "models": [
    {
      "id": 1,
      "name": "GPT-2 Base",
      "base_model": "gpt2",
      "status": "AVAILABLE",
      "description": "Standard GPT-2 model for general text generation",
      "model_size": "124M",
      "created_at": "2025-06-24T13:19:00.369561",
      "updated_at": "2025-06-24T13:19:00.369561"
    }
  ]
}
```

#### Get Model
**GET** `/api/models/{model_id}`

Get specific model information.

**Response:**
```json
{
  "id": 1,
  "name": "GPT-2 Base",
  "base_model": "gpt2",
  "status": "AVAILABLE",
  "description": "Standard GPT-2 model for general text generation",
  "model_size": "124M",
  "created_at": "2025-06-24T13:19:00.369561",
  "updated_at": "2025-06-24T13:19:00.369561",
  "parameters": null
}
```

#### Create Model
**POST** `/api/models`

Create a new model.

**Request Body:**
```json
{
  "name": "My Custom Model",
  "base_model": "gpt2",
  "description": "Custom model for specific use case",
  "model_size": "124M"
}
```

**Response:**
```json
{
  "id": 4,
  "name": "My Custom Model",
  "base_model": "gpt2",
  "status": "AVAILABLE",
  "message": "Model created successfully"
}
```

#### Generate Text
**POST** `/api/models/{model_id}/generate`

Generate text using a specific model.

**Request Body:**
```json
{
  "prompt": "The future of artificial intelligence is",
  "temperature": 0.7,
  "max_length": 100,
  "top_p": 0.9,
  "top_k": 50
}
```

**Response:**
```json
{
  "text": "The future of artificial intelligence is bright and full of possibilities...",
  "model_id": 1,
  "model_name": "GPT-2 Base",
  "generation_time": 1.23,
  "parameters": {
    "temperature": 0.7,
    "max_length": 100,
    "top_p": 0.9,
    "top_k": 50
  },
  "timestamp": "2025-06-24T13:45:30.123456"
}
```

### Training

#### List Training Jobs
**GET** `/api/training`

Get all training jobs.

**Response:**
```json
{
  "training_jobs": [
    {
      "id": 1,
      "model_id": 1,
      "job_name": "Demo LoRA Training",
      "status": "RUNNING",
      "progress": 45.5,
      "epochs": 5,
      "current_epoch": 2,
      "learning_rate": 0.0001,
      "batch_size": 8,
      "created_at": "2025-06-24T13:19:00.369561",
      "started_at": "2025-06-24T13:19:05.123456",
      "completed_at": null
    }
  ]
}
```

#### Create Training Job
**POST** `/api/training`

Create and start a new training job.

**Request Body:**
```json
{
  "model_id": 1,
  "job_name": "My Training Job",
  "epochs": 3,
  "learning_rate": 0.0001,
  "batch_size": 8,
  "lora_r": 8,
  "lora_alpha": 32,
  "lora_dropout": 0.05
}
```

**Response:**
```json
{
  "id": 4,
  "job_name": "My Training Job",
  "model_id": 1,
  "status": "PENDING",
  "message": "Training job created and started successfully"
}
```

#### Get Training Job
**GET** `/api/training/{job_id}`

Get detailed information about a specific training job.

**Response:**
```json
{
  "id": 1,
  "model_id": 1,
  "job_name": "Demo LoRA Training",
  "status": "RUNNING",
  "progress": 45.5,
  "epochs": 5,
  "current_epoch": 2,
  "learning_rate": 0.0001,
  "batch_size": 8,
  "lora_r": 8,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "current_loss": 2.45,
  "created_at": "2025-06-24T13:19:00.369561",
  "started_at": "2025-06-24T13:19:05.123456",
  "completed_at": null,
  "logs": "Epoch 1/5 completed with loss 3.21\nEpoch 2/5 in progress..."
}
```

#### Pause Training Job
**POST** `/api/training/{job_id}/pause`

Pause a running training job.

**Response:**
```json
{
  "id": 1,
  "status": "PAUSED",
  "message": "Training job paused successfully"
}
```

#### Resume Training Job
**POST** `/api/training/{job_id}/resume`

Resume a paused training job.

**Response:**
```json
{
  "id": 1,
  "status": "RUNNING",
  "message": "Training job resumed successfully"
}
```

### Evaluation

#### List Evaluations
**GET** `/api/evaluations`

Get all model evaluations.

**Response:**
```json
{
  "evaluations": [
    {
      "id": 1,
      "model_id": 1,
      "eval_name": "Standard Evaluation",
      "perplexity": 39.87,
      "bleu_score": 0.622,
      "rouge_score": 0.76,
      "response_diversity": 0.824,
      "avg_response_length": 68.3,
      "created_at": "2025-06-24T13:19:00.369561"
    }
  ]
}
```

#### Evaluate Model
**POST** `/api/models/{model_id}/evaluate`

Run evaluation on a specific model.

**Request Body (optional):**
```json
{
  "eval_name": "Custom Evaluation"
}
```

**Response:**
```json
{
  "id": 2,
  "model_id": 1,
  "eval_name": "Custom Evaluation",
  "metrics": {
    "perplexity": 39.87,
    "bleu_score": 0.622,
    "rouge_score": 0.76,
    "response_diversity": 0.824,
    "avg_response_length": 68.3
  },
  "message": "Evaluation completed successfully"
}
```

### Statistics

#### Get Platform Statistics
**GET** `/api/statistics`

Get overall platform statistics.

**Response:**
```json
{
  "total_models": 3,
  "total_training_jobs": 5,
  "running_training_jobs": 1,
  "completed_training_jobs": 2,
  "total_evaluations": 4,
  "total_generations": 15,
  "timestamp": "2025-06-24T13:45:30.123456"
}
```

## Usage Examples

### Python Example
```python
import requests

BASE_URL = "http://localhost:5000/api"

# List models
response = requests.get(f"{BASE_URL}/models")
models = response.json()['models']

# Generate text
model_id = models[0]['id']
gen_data = {
    "prompt": "Hello, AI!",
    "temperature": 0.7,
    "max_length": 50
}
response = requests.post(f"{BASE_URL}/models/{model_id}/generate", json=gen_data)
result = response.json()
print(f"Generated: {result['text']}")

# Create training job
training_data = {
    "model_id": model_id,
    "job_name": "My Training",
    "epochs": 3
}
response = requests.post(f"{BASE_URL}/training", json=training_data)
job = response.json()
print(f"Training job created: {job['id']}")
```

### curl Examples
```bash
# Health check
curl http://localhost:5000/api/health

# List models
curl http://localhost:5000/api/models

# Generate text
curl -X POST http://localhost:5000/api/models/1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "temperature": 0.7}'

# Create training job
curl -X POST http://localhost:5000/api/training \
  -H "Content-Type: application/json" \
  -d '{"model_id": 1, "job_name": "API Training", "epochs": 3}'
```