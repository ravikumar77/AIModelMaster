from app import db
from datetime import datetime
from sqlalchemy import Enum
import enum

class ModelStatus(enum.Enum):
    AVAILABLE = "AVAILABLE"
    TRAINING = "TRAINING"
    FINE_TUNING = "FINE_TUNING"
    EXPORTING = "EXPORTING"
    ERROR = "ERROR"

class TrainingStatus(enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"

class LLMModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    base_model = db.Column(db.String(128), nullable=False)
    status = db.Column(Enum(ModelStatus), default=ModelStatus.AVAILABLE)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    description = db.Column(db.Text)
    model_size = db.Column(db.String(64))
    parameters = db.Column(db.Text)  # JSON string for configuration
    
    # Relationships
    training_jobs = db.relationship('TrainingJob', backref='model', lazy=True)
    evaluations = db.relationship('Evaluation', backref='model', lazy=True)

class TrainingJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('llm_model.id'), nullable=False)
    job_name = db.Column(db.String(128), nullable=False)
    status = db.Column(Enum(TrainingStatus), default=TrainingStatus.PENDING)
    progress = db.Column(db.Float, default=0.0)
    epochs = db.Column(db.Integer, default=3)
    learning_rate = db.Column(db.Float, default=0.0001)
    batch_size = db.Column(db.Integer, default=8)
    lora_r = db.Column(db.Integer, default=8)
    lora_alpha = db.Column(db.Integer, default=32)
    lora_dropout = db.Column(db.Float, default=0.05)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    logs = db.Column(db.Text)
    current_loss = db.Column(db.Float)
    current_epoch = db.Column(db.Integer, default=0)

class Evaluation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('llm_model.id'), nullable=False)
    eval_name = db.Column(db.String(128), nullable=False)
    perplexity = db.Column(db.Float)
    bleu_score = db.Column(db.Float)
    rouge_score = db.Column(db.Float)
    response_diversity = db.Column(db.Float)
    avg_response_length = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    eval_data = db.Column(db.Text)  # JSON string for detailed metrics

class GenerationLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('llm_model.id'), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    temperature = db.Column(db.Float, default=0.7)
    max_length = db.Column(db.Integer, default=100)
    top_p = db.Column(db.Float, default=0.9)
    top_k = db.Column(db.Integer, default=50)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    generation_time = db.Column(db.Float)  # Time in seconds
