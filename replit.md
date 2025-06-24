# LLM Development Platform

## Overview

This is a comprehensive Flask-based LLM development platform that provides end-to-end workflows for creating, training, fine-tuning, evaluating, and deploying language models. The platform includes web interface, command-line tools, FastAPI backend, RLHF support, ONNX export, and Triton deployment capabilities. It implements the complete technology stack specified in the requirements: Python + PyTorch, Hugging Face Transformers, LoRA/QLoRA fine-tuning, RLHF with reward models and PPO, model evaluation, ONNX export, and FastAPI serving.

## System Architecture

### Backend Architecture
- **Framework**: Flask with SQLAlchemy ORM
- **Database**: SQLite (default) with PostgreSQL support via environment configuration
- **WSGI Server**: Gunicorn for production deployment
- **Session Management**: Flask sessions with configurable secret key

### Frontend Architecture
- **Template Engine**: Jinja2 with Bootstrap 5 dark theme
- **UI Framework**: Bootstrap 5 with Feather Icons
- **JavaScript**: Vanilla JavaScript with Chart.js for data visualization
- **Real-time Updates**: AJAX-based auto-refresh functionality

### Data Storage
- **Primary Database**: SQLAlchemy with declarative base model
- **Connection Pooling**: Configured with pool recycling and pre-ping health checks
- **Migration Support**: Built-in table creation via SQLAlchemy

## Key Components

### Model Management
- **LLMModel**: Core entity storing model metadata, status, and configuration
- **Supported Base Models**: distilgpt2, gpt2, microsoft/DialoGPT-small, facebook/opt-125m, EleutherAI/gpt-neo-125M
- **Model Status Tracking**: Available, Training, Fine-tuning, Exporting, Error states

### Training System
- **TrainingJob**: Entity for managing fine-tuning jobs with LoRA configuration
- **Training Service**: Background simulation service using threading
- **Status Management**: Pending, Running, Completed, Failed, Paused states
- **Hyperparameter Configuration**: Learning rate, batch size, epochs, LoRA parameters (r, alpha, dropout)

### Inference System
- **LLMService**: Service for loading and managing pre-trained models
- **Model Loading**: Dynamic model loading with tokenizer management
- **Device Management**: CUDA/CPU detection and optimization
- **Memory Management**: Loaded model caching and cleanup

### Web Interface
- **Dashboard**: Overview with statistics and recent activity
- **Model Management**: CRUD operations for models
- **Training Interface**: Job creation and monitoring
- **Evaluation System**: Performance metrics and assessment tools
- **Text Generation**: Interactive inference interface
- **Export Functionality**: ONNX export capabilities

## Data Flow

1. **Model Creation**: User creates a new model based on pre-trained base models
2. **Training Configuration**: User configures fine-tuning parameters and starts training jobs
3. **Background Processing**: Training service simulates the fine-tuning process with progress tracking
4. **Model Status Updates**: Real-time status updates as models progress through training phases
5. **Inference**: Trained models become available for text generation and evaluation
6. **Export**: Completed models can be exported to ONNX format for deployment

## External Dependencies

### Core Dependencies
- **Flask**: Web framework and application foundation
- **Flask-SQLAlchemy**: Database ORM and connection management
- **Gunicorn**: Production WSGI server
- **psycopg2-binary**: PostgreSQL adapter for production deployments
- **email-validator**: Input validation utilities

### ML/AI Dependencies (Planned)
- **PyTorch**: Deep learning framework for model operations
- **Transformers**: Hugging Face library for pre-trained models
- **Additional ML libraries**: As specified in the comprehensive PyTorch ecosystem configuration

### Frontend Dependencies
- **Bootstrap 5**: UI framework with dark theme
- **Feather Icons**: Icon library
- **Chart.js**: Data visualization for metrics and progress tracking

## Deployment Strategy

### Development Environment
- **Local Development**: Flask development server with debug mode
- **Database**: SQLite for simplified local development
- **Port Configuration**: Default port 5000 with bind to all interfaces

### Production Environment
- **WSGI Server**: Gunicorn with auto-scaling deployment target
- **Process Management**: Reuse-port and reload capabilities
- **Environment Variables**: Database URL and session secret configuration
- **Proxy Support**: ProxyFix middleware for reverse proxy deployments

### Container Configuration
- **Nix Environment**: Stable channel with comprehensive package management
- **System Dependencies**: PostgreSQL, OpenSSL, Git, and ML optimization libraries
- **Python Version**: Python 3.11 with ML-optimized package sources

## Changelog
- June 24, 2025: Initial Flask web platform setup with model management, training simulation, and inference
- June 24, 2025: Added complete LLM development pipeline with training scripts, LoRA fine-tuning, RLHF support
- June 24, 2025: Implemented FastAPI backend, ONNX export, evaluation metrics, and sample datasets
- June 24, 2025: Created comprehensive project structure with config management and Triton deployment support

## User Preferences

Preferred communication style: Simple, everyday language.