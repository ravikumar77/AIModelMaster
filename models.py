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


# Prompt Playground Models
class PromptTemplate(db.Model):
    __tablename__ = 'prompt_template'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    template_content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(64), default='general')  # general, coding, creative, etc.
    tags = db.Column(db.Text)  # JSON array of tags
    is_public = db.Column(db.Boolean, default=False)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    usage_count = db.Column(db.Integer, default=0)
    
    # Relationships
    prompt_sessions = db.relationship('PromptSession', backref='template', lazy=True)
    
    def __repr__(self):
        return f'<PromptTemplate {self.name}>'


class PromptSession(db.Model):
    __tablename__ = 'prompt_session'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    prompt_text = db.Column(db.Text, nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey('llm_model.id'), nullable=False)
    template_id = db.Column(db.Integer, db.ForeignKey('prompt_template.id'), nullable=True)
    
    # Generation parameters
    temperature = db.Column(db.Float, default=0.7)
    max_length = db.Column(db.Integer, default=100)
    top_p = db.Column(db.Float, default=0.9)
    top_k = db.Column(db.Integer, default=50)
    repetition_penalty = db.Column(db.Float, default=1.0)
    
    # Session metadata
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_favorite = db.Column(db.Boolean, default=False)
    tags = db.Column(db.Text)  # JSON array of tags
    
    # Chat context
    context_messages = db.Column(db.Text)  # JSON array of previous messages
    few_shot_examples = db.Column(db.Text)  # JSON array of examples
    
    # Relationships
    generations = db.relationship('PromptGeneration', backref='session', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<PromptSession {self.name}>'


class PromptGeneration(db.Model):
    __tablename__ = 'prompt_generation'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('prompt_session.id'), nullable=False)
    
    # Input/Output
    input_text = db.Column(db.Text, nullable=False)
    generated_text = db.Column(db.Text, nullable=False)
    full_prompt = db.Column(db.Text, nullable=False)  # Complete prompt sent to model
    
    # Generation metadata
    generation_time = db.Column(db.Float)  # Time in seconds
    tokens_generated = db.Column(db.Integer)
    tokens_per_second = db.Column(db.Float)
    
    # Used parameters (may differ from session defaults)
    temperature = db.Column(db.Float)
    max_length = db.Column(db.Integer)
    top_p = db.Column(db.Float)
    top_k = db.Column(db.Integer)
    
    # Quality metrics
    user_rating = db.Column(db.Integer)  # 1-5 stars
    is_flagged = db.Column(db.Boolean, default=False)
    flag_reason = db.Column(db.String(256))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<PromptGeneration {self.id}>'


class PromptExport(db.Model):
    __tablename__ = 'prompt_export'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('prompt_session.id'), nullable=False)
    export_format = db.Column(db.String(32), nullable=False)  # json, yaml, curl, python
    export_content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    download_count = db.Column(db.Integer, default=0)
    
    def __repr__(self):
        return f'<PromptExport {self.export_format}>'


# Experiment Tracking Models
class ExperimentStatus(enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class Experiment(db.Model):
    __tablename__ = 'experiment'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    status = db.Column(Enum(ExperimentStatus), default=ExperimentStatus.PENDING)
    
    # Associated training job
    training_job_id = db.Column(db.Integer, db.ForeignKey('training_job.id'), nullable=True)
    model_id = db.Column(db.Integer, db.ForeignKey('llm_model.id'), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('coding_dataset.id'), nullable=True)
    
    # Experiment configuration
    hyperparameters = db.Column(db.Text)  # JSON string
    runtime_settings = db.Column(db.Text)  # JSON string
    dataset_config = db.Column(db.Text)  # JSON string
    
    # Metadata
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    
    # Organization
    tags = db.Column(db.Text)  # JSON array of tags
    is_favorite = db.Column(db.Boolean, default=False)
    is_archived = db.Column(db.Boolean, default=False)
    experiment_group = db.Column(db.String(128))  # For grouping related experiments
    
    # Hardware/Resource tracking
    gpu_hours = db.Column(db.Float, default=0.0)
    estimated_cost = db.Column(db.Float, default=0.0)
    memory_peak_gb = db.Column(db.Float)
    disk_usage_gb = db.Column(db.Float)
    
    # Relationships
    metrics = db.relationship('ExperimentMetric', backref='experiment', lazy=True, cascade='all, delete-orphan')
    comparisons = db.relationship('ExperimentComparison', backref='experiment', lazy=True)
    artifacts = db.relationship('ExperimentArtifact', backref='experiment', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Experiment {self.name}>'


class ExperimentMetric(db.Model):
    __tablename__ = 'experiment_metric'
    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(db.Integer, db.ForeignKey('experiment.id'), nullable=False)
    
    # Metric details
    metric_name = db.Column(db.String(64), nullable=False)  # loss, perplexity, bleu, rouge, etc.
    metric_type = db.Column(db.String(32), nullable=False)  # training, validation, test
    value = db.Column(db.Float, nullable=False)
    
    # Time series data
    epoch = db.Column(db.Integer)
    step = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Additional metadata
    metric_metadata = db.Column(db.Text)  # JSON for additional info
    
    def __repr__(self):
        return f'<ExperimentMetric {self.metric_name}:{self.value}>'


class ExperimentComparison(db.Model):
    __tablename__ = 'experiment_comparison'
    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(db.Integer, db.ForeignKey('experiment.id'), nullable=False)
    comparison_name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    
    # Comparison data
    compared_experiments = db.Column(db.Text)  # JSON array of experiment IDs
    comparison_metrics = db.Column(db.Text)  # JSON array of metrics to compare
    comparison_results = db.Column(db.Text)  # JSON comparison analysis
    
    # Visualization settings
    chart_config = db.Column(db.Text)  # JSON chart configuration
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    def __repr__(self):
        return f'<ExperimentComparison {self.comparison_name}>'


class ExperimentArtifact(db.Model):
    __tablename__ = 'experiment_artifact'
    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(db.Integer, db.ForeignKey('experiment.id'), nullable=False)
    
    # Artifact details
    artifact_name = db.Column(db.String(128), nullable=False)
    artifact_type = db.Column(db.String(64), nullable=False)  # model, logs, plots, data
    file_path = db.Column(db.String(512))
    file_size = db.Column(db.BigInteger)  # Size in bytes
    
    # Content metadata
    description = db.Column(db.Text)
    artifact_metadata = db.Column(db.Text)  # JSON metadata
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ExperimentArtifact {self.artifact_name}>'


class ExperimentNote(db.Model):
    __tablename__ = 'experiment_note'
    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(db.Integer, db.ForeignKey('experiment.id'), nullable=False)
    
    # Note content
    title = db.Column(db.String(128))
    content = db.Column(db.Text, nullable=False)
    note_type = db.Column(db.String(32), default='general')  # general, hypothesis, observation, conclusion
    
    # Organization
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    def __repr__(self):
        return f'<ExperimentNote {self.title}>'
