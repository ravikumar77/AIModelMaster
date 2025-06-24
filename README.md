# LLM Development Platform

A comprehensive web-based platform for developing, training, fine-tuning, and deploying Large Language Models (LLMs) using modern AI technologies.

## Features

### 🚀 Complete LLM Workflow
- **Model Management**: Create and manage language models based on popular pre-trained models
- **Training Pipeline**: Full training from scratch with configurable hyperparameters
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning using LoRA/QLoRA
- **RLHF Support**: Reinforcement Learning with Human Feedback for model alignment
- **Model Evaluation**: Comprehensive metrics including perplexity, BLEU, ROUGE, and diversity
- **ONNX Export**: Export models for optimized inference and deployment
- **Text Generation**: Interactive interface for testing model outputs

### 🛠 Technology Stack
- **Backend**: Flask with SQLAlchemy ORM
- **Frontend**: Bootstrap 5 with dark theme, Feather Icons, Chart.js
- **ML Libraries**: PyTorch, Transformers, PEFT, TRL, ONNX
- **Training**: LoRA/QLoRA fine-tuning, PPO for RLHF
- **Deployment**: Gunicorn WSGI server, Triton Inference Server support
- **API**: FastAPI backend for programmatic access

### 🎯 Key Components

#### Web Interface
- **Dashboard**: Overview with statistics and recent activity
- **Model Management**: CRUD operations for language models
- **Training Interface**: Job creation and real-time monitoring
- **Inference Engine**: Interactive text generation with parameter controls
- **Evaluation Suite**: Performance metrics and benchmarking tools
- **Export Tools**: ONNX conversion with optimization options

#### Command Line Tools
- `scripts/train.py`: Full model training from scratch
- `scripts/fine_tune_lora.py`: LoRA-based fine-tuning
- `scripts/generate.py`: Command-line text generation
- `scripts/evaluate.py`: Model evaluation and benchmarking
- `scripts/export_to_onnx.py`: ONNX export with optimizations
- `scripts/rlhf/reward_model.py`: Reward model training
- `scripts/rlhf/ppo_trainer.py`: PPO training for RLHF

#### API Endpoints
- **Model Management**: List, create, and manage models
- **Text Generation**: Synchronous and streaming inference
- **Training Control**: Start, monitor, and control training jobs
- **Evaluation**: Run and retrieve evaluation results
- **Export**: ONNX conversion and optimization

## Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (optional, for faster training)

### Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Flask application:
   ```bash
   gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
   ```
4. Access the web interface at `http://localhost:5000`

### Optional: Start FastAPI backend
```bash
cd api && python app.py
```
API will be available at `http://localhost:8000`

## Quick Start

### 1. Create a Model
- Navigate to Models → Create New Model
- Select a base model (GPT-2, DistilGPT-2, etc.)
- Provide a name and description

### 2. Fine-tune with LoRA
- Go to Training → Start New Training
- Select your model and configure LoRA parameters
- Monitor progress in real-time

### 3. Generate Text
- Visit Inference → Generate Text
- Select your trained model
- Adjust generation parameters and test outputs

### 4. Evaluate Performance
- Use Evaluation → Run Evaluation
- Get comprehensive metrics including BLEU, ROUGE, and perplexity

### 5. Export for Deployment
- Go to Export → ONNX Export
- Choose optimization level (basic, optimized, quantized)
- Download optimized model for deployment

## Configuration

### Model Configuration (`config/model_config.yaml`)
- Base model settings and supported architectures
- Training hyperparameters and limits
- LoRA configuration options
- Evaluation metrics and benchmarks
- Export optimization settings

### Environment Variables
- `DATABASE_URL`: Database connection string
- `SESSION_SECRET`: Flask session secret key

## Project Structure

```
llm_project/
├── config/
│   └── model_config.yaml
├── data/
│   ├── sample_training_data.json
│   ├── sample_preference_data.json
│   └── evaluation_prompts.json
├── models/
│   ├── base/              # Base models
│   ├── finetuned/         # Fine-tuned models
│   └── onnx/              # Exported ONNX models
├── scripts/
│   ├── train.py           # Full training
│   ├── fine_tune_lora.py  # LoRA fine-tuning
│   ├── generate.py        # Text generation
│   ├── evaluate.py        # Model evaluation
│   ├── export_to_onnx.py  # ONNX export
│   └── rlhf/
│       ├── reward_model.py
│       └── ppo_trainer.py
├── api/
│   └── app.py             # FastAPI backend
├── triton_config/
│   └── config.pbtxt       # Triton configuration
├── templates/             # HTML templates
├── static/                # CSS, JS, assets
├── app.py                 # Flask application
├── models.py              # Database models
├── routes.py              # Web routes
├── llm_service.py         # LLM inference service
├── training_service.py    # Training management
└── main.py               # Application entry point
```

## Usage Examples

### Command Line Training
```bash
# Full training
python scripts/train.py --model gpt2 --data data/sample_training_data.json --epochs 3

# LoRA fine-tuning
python scripts/fine_tune_lora.py --model gpt2 --data data/sample_training_data.json --lora-r 8

# Text generation
python scripts/generate.py --model models/finetuned/my_model --prompt "Explain AI" --interactive

# Model evaluation
python scripts/evaluate.py --model models/finetuned/my_model --dataset data/evaluation_prompts.json

# ONNX export
python scripts/export_to_onnx.py --model models/finetuned/my_model --optimization quantized
```

### API Usage
```python
import requests

# Generate text
response = requests.post("http://localhost:8000/generate", json={
    "model_id": 1,
    "prompt": "Explain artificial intelligence",
    "temperature": 0.7,
    "max_length": 100
})

# Start training
response = requests.post("http://localhost:8000/training/start", json={
    "model_id": 1,
    "job_name": "Fine-tune GPT-2",
    "epochs": 3,
    "lora_r": 8
})
```

## Advanced Features

### RLHF (Reinforcement Learning with Human Feedback)
1. Train a reward model on preference data:
   ```bash
   python scripts/rlhf/reward_model.py --data data/sample_preference_data.json
   ```

2. Use PPO to align the model:
   ```bash
   python scripts/rlhf/ppo_trainer.py --model gpt2 --reward-model models/reward_model --prompts data/prompts.json
   ```

### Triton Inference Server
Deploy ONNX models with Triton for high-performance inference:
1. Export model to ONNX format
2. Use generated `triton_config/config.pbtxt`
3. Deploy with Triton Inference Server

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Check the documentation in `/docs`
- Review sample configurations in `/config`
- Examine example data in `/data`
- Use the web interface help sections

## Roadmap

- [ ] Docker containerization
- [ ] Distributed training support
- [ ] Model versioning and registry
- [ ] Advanced evaluation metrics
- [ ] Integration with Weights & Biases
- [ ] Auto-scaling deployment options
- [ ] Custom tokenizer support
- [ ] Multi-modal model support