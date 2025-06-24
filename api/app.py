#!/usr/bin/env python3
"""
FastAPI backend for LLM inference and model management
Provides REST API endpoints for text generation and model operations
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    print("FastAPI not available")
    FASTAPI_AVAILABLE = False

from app import app as flask_app, db
from models import LLMModel, GenerationLog, TrainingJob, Evaluation
from llm_service import LLMService
from training_service import TrainingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
api_app = FastAPI(
    title="LLM Development Platform API",
    description="REST API for LLM training, inference, and management",
    version="1.0.0"
)

# CORS middleware
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services
llm_service = LLMService()
training_service = TrainingService()

# Pydantic models
class PromptRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for generation")
    model_id: int = Field(..., description="ID of the model to use")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Generation temperature")
    max_length: int = Field(100, ge=10, le=1000, description="Maximum generation length")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Top-p sampling parameter")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling parameter")
    stream: bool = Field(False, description="Stream response tokens")

class GenerationResponse(BaseModel):
    text: str
    model_id: int
    generation_time: float
    tokens_generated: Optional[int] = None
    timestamp: datetime

class ModelInfo(BaseModel):
    id: int
    name: str
    base_model: str
    status: str
    created_at: datetime
    model_size: Optional[str] = None

class TrainingRequest(BaseModel):
    model_id: int
    job_name: str
    epochs: int = Field(3, ge=1, le=20)
    learning_rate: float = Field(0.0001, ge=0.00001, le=0.01)
    batch_size: int = Field(8, ge=1, le=32)
    lora_r: int = Field(8, ge=1, le=64)
    lora_alpha: int = Field(32, ge=1, le=128)
    lora_dropout: float = Field(0.05, ge=0.0, le=0.5)
    data_path: Optional[str] = None

class EvaluationRequest(BaseModel):
    model_id: int
    eval_name: str
    dataset_path: Optional[str] = None

# Health check
@api_app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "services": {
            "llm_service": "available",
            "training_service": "available",
            "database": "connected"
        }
    }

# Model endpoints
@api_app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models"""
    with flask_app.app_context():
        models = LLMModel.query.all()
        return [
            ModelInfo(
                id=model.id,
                name=model.name,
                base_model=model.base_model,
                status=model.status.value,
                created_at=model.created_at,
                model_size=model.model_size
            )
            for model in models
        ]

@api_app.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: int):
    """Get specific model information"""
    with flask_app.app_context():
        model = LLMModel.query.get(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return ModelInfo(
            id=model.id,
            name=model.name,
            base_model=model.base_model,
            status=model.status.value,
            created_at=model.created_at,
            model_size=model.model_size
        )

# Generation endpoints
@api_app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: PromptRequest):
    """Generate text using specified model"""
    try:
        start_time = datetime.now()
        
        # Generate text
        response_text = llm_service.generate_text(
            model_id=request.model_id,
            prompt=request.prompt,
            temperature=request.temperature,
            max_length=request.max_length,
            top_p=request.top_p,
            top_k=request.top_k
        )
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # Log generation
        with flask_app.app_context():
            log = GenerationLog(
                model_id=request.model_id,
                prompt=request.prompt,
                response=response_text,
                temperature=request.temperature,
                max_length=request.max_length,
                top_p=request.top_p,
                top_k=request.top_k,
                generation_time=generation_time
            )
            db.session.add(log)
            db.session.commit()
        
        return GenerationResponse(
            text=response_text,
            model_id=request.model_id,
            generation_time=generation_time,
            tokens_generated=len(response_text.split()),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_app.post("/generate/stream")
async def generate_text_stream(request: PromptRequest):
    """Stream text generation"""
    async def generate():
        try:
            # For streaming, we'll simulate token-by-token generation
            response_text = llm_service.generate_text(
                model_id=request.model_id,
                prompt=request.prompt,
                temperature=request.temperature,
                max_length=request.max_length,
                top_p=request.top_p,
                top_k=request.top_k
            )
            
            # Simulate streaming by yielding words
            words = response_text.split()
            for i, word in enumerate(words):
                chunk = {
                    "text": word + " ",
                    "finished": i == len(words) - 1
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                
        except Exception as e:
            error_chunk = {"error": str(e)}
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/plain")

# Training endpoints
@api_app.post("/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a training job"""
    try:
        with flask_app.app_context():
            # Create training job
            job = TrainingJob(
                model_id=request.model_id,
                job_name=request.job_name,
                epochs=request.epochs,
                learning_rate=request.learning_rate,
                batch_size=request.batch_size,
                lora_r=request.lora_r,
                lora_alpha=request.lora_alpha,
                lora_dropout=request.lora_dropout
            )
            
            db.session.add(job)
            db.session.commit()
            
            # Start training in background
            background_tasks.add_task(training_service.start_training_simulation, job.id)
            
            return {
                "job_id": job.id,
                "status": "started",
                "message": f"Training job '{request.job_name}' started successfully"
            }
            
    except Exception as e:
        logger.error(f"Training start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_app.get("/training/{job_id}/status")
async def get_training_status(job_id: int):
    """Get training job status"""
    with flask_app.app_context():
        job = TrainingJob.query.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        return {
            "job_id": job.id,
            "status": job.status.value,
            "progress": job.progress,
            "current_epoch": job.current_epoch,
            "current_loss": job.current_loss,
            "logs": job.logs,
            "started_at": job.started_at,
            "completed_at": job.completed_at
        }

@api_app.post("/training/{job_id}/pause")
async def pause_training(job_id: int):
    """Pause a running training job"""
    try:
        training_service.pause_training(job_id)
        return {"message": f"Training job {job_id} paused"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_app.post("/training/{job_id}/resume")
async def resume_training(job_id: int):
    """Resume a paused training job"""
    try:
        training_service.resume_training(job_id)
        return {"message": f"Training job {job_id} resumed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Evaluation endpoints
@api_app.post("/evaluation/run")
async def run_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Run model evaluation"""
    try:
        with flask_app.app_context():
            # Run evaluation in background
            metrics = training_service.run_evaluation(request.model_id)
            
            # Save evaluation results
            evaluation = Evaluation(
                model_id=request.model_id,
                eval_name=request.eval_name,
                perplexity=metrics['perplexity'],
                bleu_score=metrics['bleu_score'],
                rouge_score=metrics['rouge_score'],
                response_diversity=metrics['response_diversity'],
                avg_response_length=metrics['avg_response_length']
            )
            
            db.session.add(evaluation)
            db.session.commit()
            
            return {
                "evaluation_id": evaluation.id,
                "metrics": metrics,
                "message": f"Evaluation '{request.eval_name}' completed"
            }
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_app.get("/evaluation/{model_id}")
async def get_evaluations(model_id: int):
    """Get evaluation results for a model"""
    with flask_app.app_context():
        evaluations = Evaluation.query.filter_by(model_id=model_id).all()
        
        return [
            {
                "id": eval.id,
                "eval_name": eval.eval_name,
                "perplexity": eval.perplexity,
                "bleu_score": eval.bleu_score,
                "rouge_score": eval.rouge_score,
                "response_diversity": eval.response_diversity,
                "avg_response_length": eval.avg_response_length,
                "created_at": eval.created_at
            }
            for eval in evaluations
        ]

# Export endpoints
@api_app.post("/export/onnx/{model_id}")
async def export_to_onnx(model_id: int, background_tasks: BackgroundTasks):
    """Export model to ONNX format"""
    try:
        with flask_app.app_context():
            model = LLMModel.query.get(model_id)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Start export in background
            background_tasks.add_task(training_service.simulate_export, model_id)
            
            return {
                "message": f"ONNX export started for model '{model.name}'",
                "model_id": model_id
            }
            
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Statistics endpoints
@api_app.get("/stats")
async def get_statistics():
    """Get platform statistics"""
    with flask_app.app_context():
        total_models = LLMModel.query.count()
        total_jobs = TrainingJob.query.count()
        total_generations = GenerationLog.query.count()
        total_evaluations = Evaluation.query.count()
        
        training_stats = training_service.get_training_statistics()
        
        return {
            "models": {
                "total": total_models,
                "by_status": {
                    "available": LLMModel.query.filter_by(status="available").count(),
                    "training": LLMModel.query.filter_by(status="training").count(),
                    "error": LLMModel.query.filter_by(status="error").count()
                }
            },
            "training": training_stats,
            "generations": total_generations,
            "evaluations": total_evaluations
        }

def main():
    """Run the FastAPI server"""
    if not FASTAPI_AVAILABLE:
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
        return
    
    uvicorn.run(
        "api.app:api_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()