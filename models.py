from datetime import datetime
from sqlalchemy import Enum
import enum
from app import db

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
    __tablename__ = 'llm_model'
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
    __tablename__ = 'training_job'
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
    dataset_id = db.Column(db.Integer, db.ForeignKey('coding_dataset.id'), nullable=True)
    training_type = db.Column(db.String(64), default='general')  # general, coding, instruction_following

class Evaluation(db.Model):
    __tablename__ = 'evaluation'
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
    __tablename__ = 'generation_log'
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
    api_key_id = db.Column(db.Integer, db.ForeignKey('api_key.id'), nullable=True)
    
    # Relationship to access model information
    model = db.relationship('LLMModel', backref='generation_logs')

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    api_keys = db.relationship('APIKey', backref='user', lazy=True)

class APIKey(db.Model):
    __tablename__ = 'api_key'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    key_name = db.Column(db.String(128), nullable=False)
    key_value = db.Column(db.String(256), unique=True, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_used = db.Column(db.DateTime)
    usage_count = db.Column(db.Integer, default=0)
    rate_limit = db.Column(db.Integer, default=1000)  # requests per day
    
    def __repr__(self):
        return f'<APIKey {self.key_name}>'

class CodingDataset(db.Model):
    __tablename__ = 'coding_dataset'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    language = db.Column(db.String(64))  # Python, JavaScript, etc.
    dataset_type = db.Column(db.String(64))  # code_completion, bug_fixing, etc.
    file_path = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<CodingDataset {self.name}>'
