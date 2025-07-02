"""
Experiment Tracking & Comparison Service - Advanced ML experiment management
"""
import json
import logging
import numpy as np
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
# Use Flask's app context for database operations
def get_db():
    """Get database session"""
    from app import db
    return db
from models import (
    Experiment, ExperimentMetric, ExperimentComparison, ExperimentArtifact, 
    ExperimentNote, ExperimentStatus, TrainingJob, LLMModel, CodingDataset
)

logger = logging.getLogger(__name__)

class ExperimentTrackingService:
    def __init__(self):
        self.supported_metrics = [
            'loss', 'val_loss', 'perplexity', 'val_perplexity',
            'bleu_score', 'rouge_1', 'rouge_2', 'rouge_l',
            'lambada_accuracy', 'learning_rate', 'gradient_norm',
            'gpu_memory', 'training_speed', 'accuracy'
        ]
        
    # Experiment Management
    def create_experiment(self, name: str, description: str, model_id: int,
                         hyperparameters: Dict[str, Any], dataset_id: int = None,
                         runtime_settings: Dict[str, Any] = None,
                         created_by: int = None, tags: List[str] = None,
                         experiment_group: str = None) -> Experiment:
        """Create a new experiment"""
        try:
            experiment = Experiment(
                name=name,
                description=description,
                model_id=model_id,
                dataset_id=dataset_id,
                hyperparameters=json.dumps(hyperparameters),
                runtime_settings=json.dumps(runtime_settings or {}),
                created_by=created_by,
                tags=json.dumps(tags or []),
                experiment_group=experiment_group,
                status=ExperimentStatus.PENDING
            )
            
            get_db().session.add(experiment)
            get_db().session.commit()
            
            logger.info(f"Created experiment: {name}")
            return experiment
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating experiment: {e}")
            raise
    
    def start_experiment(self, experiment_id: int, training_job_id: int = None) -> bool:
        """Start an experiment and begin tracking"""
        try:
            experiment = Experiment.query.get(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
                
            experiment.status = ExperimentStatus.RUNNING
            experiment.started_at = datetime.utcnow()
            experiment.training_job_id = training_job_id
            
            db.session.commit()
            
            # Start metric collection simulation
            self._start_metric_simulation(experiment_id)
            
            logger.info(f"Started experiment: {experiment.name}")
            return True
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error starting experiment: {e}")
            return False
    
    def complete_experiment(self, experiment_id: int, final_metrics: Dict[str, float] = None) -> bool:
        """Complete an experiment and log final metrics"""
        try:
            experiment = Experiment.query.get(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
                
            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = datetime.utcnow()
            
            # Calculate resource usage
            if experiment.started_at:
                runtime_hours = (experiment.completed_at - experiment.started_at).total_seconds() / 3600
                experiment.gpu_hours = runtime_hours * 1.0  # Assume 1 GPU
                experiment.estimated_cost = runtime_hours * 2.5  # $2.5/hour for GPU
            
            # Log final metrics
            if final_metrics:
                for metric_name, value in final_metrics.items():
                    self.log_metric(experiment_id, metric_name, value, 'test')
            
            db.session.commit()
            
            logger.info(f"Completed experiment: {experiment.name}")
            return True
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error completing experiment: {e}")
            return False
    
    # Metric Logging
    def log_metric(self, experiment_id: int, metric_name: str, value: float,
                   metric_type: str = 'training', epoch: int = None,
                   step: int = None, metadata: Dict[str, Any] = None) -> bool:
        """Log a metric for an experiment"""
        try:
            metric = ExperimentMetric(
                experiment_id=experiment_id,
                metric_name=metric_name,
                metric_type=metric_type,
                value=value,
                epoch=epoch,
                step=step,
                metadata=json.dumps(metadata or {})
            )
            
            db.session.add(metric)
            db.session.commit()
            
            return True
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error logging metric: {e}")
            return False
    
    def log_metrics_batch(self, experiment_id: int, metrics: List[Dict[str, Any]]) -> bool:
        """Log multiple metrics in batch"""
        try:
            metric_objects = []
            for metric_data in metrics:
                metric = ExperimentMetric(
                    experiment_id=experiment_id,
                    metric_name=metric_data['name'],
                    metric_type=metric_data.get('type', 'training'),
                    value=metric_data['value'],
                    epoch=metric_data.get('epoch'),
                    step=metric_data.get('step'),
                    metadata=json.dumps(metric_data.get('metadata', {}))
                )
                metric_objects.append(metric)
            
            db.session.add_all(metric_objects)
            db.session.commit()
            
            return True
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error logging metrics batch: {e}")
            return False
    
    # Data Retrieval
    def get_experiments(self, user_id: int = None, status: str = None,
                       experiment_group: str = None, tags: List[str] = None,
                       limit: int = 50) -> List[Experiment]:
        """Get experiments with filtering"""
        query = Experiment.query
        
        if user_id:
            query = query.filter_by(created_by=user_id)
        if status:
            query = query.filter_by(status=status)
        if experiment_group:
            query = query.filter_by(experiment_group=experiment_group)
        if tags:
            # Filter by tags (simplified - would need proper JSON querying in production)
            for tag in tags:
                query = query.filter(Experiment.tags.contains(f'"{tag}"'))
        
        query = query.filter_by(is_archived=False)
        return query.order_by(Experiment.created_at.desc()).limit(limit).all()
    
    def get_experiment_metrics(self, experiment_id: int, metric_names: List[str] = None,
                              metric_type: str = None) -> List[ExperimentMetric]:
        """Get metrics for an experiment"""
        query = ExperimentMetric.query.filter_by(experiment_id=experiment_id)
        
        if metric_names:
            query = query.filter(ExperimentMetric.metric_name.in_(metric_names))
        if metric_type:
            query = query.filter_by(metric_type=metric_type)
            
        return query.order_by(ExperimentMetric.timestamp.asc()).all()
    
    def get_metric_timeseries(self, experiment_id: int, metric_name: str,
                             metric_type: str = 'training') -> Dict[str, List]:
        """Get time series data for a specific metric"""
        metrics = ExperimentMetric.query.filter_by(
            experiment_id=experiment_id,
            metric_name=metric_name,
            metric_type=metric_type
        ).order_by(ExperimentMetric.timestamp.asc()).all()
        
        return {
            'timestamps': [m.timestamp.isoformat() for m in metrics],
            'values': [m.value for m in metrics],
            'epochs': [m.epoch for m in metrics if m.epoch is not None],
            'steps': [m.step for m in metrics if m.step is not None]
        }
    
    # Comparison and Analysis
    def create_comparison(self, experiment_ids: List[int], comparison_name: str,
                         metrics_to_compare: List[str], created_by: int = None,
                         description: str = "") -> ExperimentComparison:
        """Create a comparison between experiments"""
        try:
            # Validate experiments exist
            experiments = Experiment.query.filter(Experiment.id.in_(experiment_ids)).all()
            if len(experiments) != len(experiment_ids):
                raise ValueError("Some experiments not found")
            
            # Generate comparison analysis
            comparison_results = self._analyze_experiments(experiment_ids, metrics_to_compare)
            
            # Create comparison record
            comparison = ExperimentComparison(
                experiment_id=experiment_ids[0],  # Primary experiment
                comparison_name=comparison_name,
                description=description,
                compared_experiments=json.dumps(experiment_ids),
                comparison_metrics=json.dumps(metrics_to_compare),
                comparison_results=json.dumps(comparison_results),
                created_by=created_by
            )
            
            db.session.add(comparison)
            db.session.commit()
            
            logger.info(f"Created comparison: {comparison_name}")
            return comparison
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating comparison: {e}")
            raise
    
    def _analyze_experiments(self, experiment_ids: List[int], 
                           metrics_to_compare: List[str]) -> Dict[str, Any]:
        """Analyze and compare experiments"""
        analysis = {
            'summary': {},
            'best_performing': {},
            'metric_comparisons': {},
            'convergence_analysis': {},
            'statistical_significance': {}
        }
        
        for metric_name in metrics_to_compare:
            metric_data = {}
            
            for exp_id in experiment_ids:
                timeseries = self.get_metric_timeseries(exp_id, metric_name)
                if timeseries['values']:
                    final_value = timeseries['values'][-1]
                    best_value = min(timeseries['values']) if 'loss' in metric_name else max(timeseries['values'])
                    convergence_epoch = self._find_convergence_point(timeseries['values'])
                    
                    metric_data[exp_id] = {
                        'final_value': final_value,
                        'best_value': best_value,
                        'convergence_epoch': convergence_epoch,
                        'stability': self._calculate_stability(timeseries['values'])
                    }
            
            if metric_data:
                # Find best performing experiment for this metric
                if 'loss' in metric_name:
                    best_exp = min(metric_data.items(), key=lambda x: x[1]['best_value'])
                else:
                    best_exp = max(metric_data.items(), key=lambda x: x[1]['best_value'])
                
                analysis['best_performing'][metric_name] = best_exp[0]
                analysis['metric_comparisons'][metric_name] = metric_data
        
        return analysis
    
    def _find_convergence_point(self, values: List[float], window_size: int = 10) -> int:
        """Find the epoch where the metric converged"""
        if len(values) < window_size * 2:
            return len(values)
        
        for i in range(window_size, len(values) - window_size):
            window1 = values[i-window_size:i]
            window2 = values[i:i+window_size]
            
            if abs(np.mean(window1) - np.mean(window2)) < 0.01:  # Convergence threshold
                return i
        
        return len(values)
    
    def _calculate_stability(self, values: List[float]) -> float:
        """Calculate stability score (lower variance = more stable)"""
        if len(values) < 2:
            return 1.0
        
        # Use coefficient of variation as stability measure
        mean_val = np.mean(values[-10:])  # Last 10 values
        std_val = np.std(values[-10:])
        
        if mean_val == 0:
            return 1.0
        
        cv = std_val / abs(mean_val)
        return max(0, 1 - cv)  # Higher score = more stable
    
    # Visualization Data
    def get_comparison_chart_data(self, comparison_id: int) -> Dict[str, Any]:
        """Get chart data for comparison visualization"""
        comparison = ExperimentComparison.query.get(comparison_id)
        if not comparison:
            raise ValueError(f"Comparison {comparison_id} not found")
        
        experiment_ids = json.loads(comparison.compared_experiments)
        metrics = json.loads(comparison.comparison_metrics)
        
        chart_data = {
            'experiments': {},
            'metrics': metrics,
            'chart_configs': []
        }
        
        # Get experiment details
        experiments = Experiment.query.filter(Experiment.id.in_(experiment_ids)).all()
        for exp in experiments:
            chart_data['experiments'][exp.id] = {
                'name': exp.name,
                'description': exp.description,
                'status': exp.status.value,
                'created_at': exp.created_at.isoformat()
            }
        
        # Generate chart configurations for each metric
        for metric in metrics:
            chart_config = {
                'metric_name': metric,
                'chart_type': 'line',
                'data': {},
                'layout': {
                    'title': f'{metric.title()} Comparison',
                    'xaxis': {'title': 'Epoch'},
                    'yaxis': {'title': metric.title()}
                }
            }
            
            for exp_id in experiment_ids:
                timeseries = self.get_metric_timeseries(exp_id, metric)
                if timeseries['values']:
                    exp_name = chart_data['experiments'][exp_id]['name']
                    chart_config['data'][exp_name] = {
                        'x': list(range(len(timeseries['values']))),
                        'y': timeseries['values']
                    }
            
            chart_data['chart_configs'].append(chart_config)
        
        return chart_data
    
    # Organization and Management
    def toggle_favorite(self, experiment_id: int) -> bool:
        """Toggle experiment favorite status"""
        experiment = Experiment.query.get(experiment_id)
        if experiment:
            experiment.is_favorite = not experiment.is_favorite
            db.session.commit()
            return experiment.is_favorite
        return False
    
    def archive_experiment(self, experiment_id: int) -> bool:
        """Archive an experiment"""
        experiment = Experiment.query.get(experiment_id)
        if experiment:
            experiment.is_archived = True
            experiment.status = ExperimentStatus.ARCHIVED
            db.session.commit()
            return True
        return False
    
    def add_tags(self, experiment_id: int, new_tags: List[str]) -> bool:
        """Add tags to an experiment"""
        experiment = Experiment.query.get(experiment_id)
        if experiment:
            existing_tags = json.loads(experiment.tags or '[]')
            updated_tags = list(set(existing_tags + new_tags))
            experiment.tags = json.dumps(updated_tags)
            db.session.commit()
            return True
        return False
    
    # Notes and Documentation
    def add_note(self, experiment_id: int, title: str, content: str,
                note_type: str = 'general', created_by: int = None) -> ExperimentNote:
        """Add a note to an experiment"""
        try:
            note = ExperimentNote(
                experiment_id=experiment_id,
                title=title,
                content=content,
                note_type=note_type,
                created_by=created_by
            )
            
            db.session.add(note)
            db.session.commit()
            
            return note
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error adding note: {e}")
            raise
    
    # Artifact Management
    def log_artifact(self, experiment_id: int, artifact_name: str, artifact_type: str,
                    file_path: str = None, description: str = "",
                    metadata: Dict[str, Any] = None) -> ExperimentArtifact:
        """Log an artifact for an experiment"""
        try:
            artifact = ExperimentArtifact(
                experiment_id=experiment_id,
                artifact_name=artifact_name,
                artifact_type=artifact_type,
                file_path=file_path,
                description=description,
                metadata=json.dumps(metadata or {})
            )
            
            db.session.add(artifact)
            db.session.commit()
            
            return artifact
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error logging artifact: {e}")
            raise
    
    # Statistics and Analytics
    def get_experiment_statistics(self) -> Dict[str, Any]:
        """Get overall experiment statistics"""
        total_experiments = Experiment.query.count()
        running_experiments = Experiment.query.filter_by(status=ExperimentStatus.RUNNING).count()
        completed_experiments = Experiment.query.filter_by(status=ExperimentStatus.COMPLETED).count()
        
        # Calculate total GPU hours and cost
        completed_exps = Experiment.query.filter_by(status=ExperimentStatus.COMPLETED).all()
        total_gpu_hours = sum(exp.gpu_hours or 0 for exp in completed_exps)
        total_cost = sum(exp.estimated_cost or 0 for exp in completed_exps)
        
        # Top performing experiments
        favorite_count = Experiment.query.filter_by(is_favorite=True).count()
        
        return {
            'total_experiments': total_experiments,
            'running_experiments': running_experiments,
            'completed_experiments': completed_experiments,
            'total_gpu_hours': round(total_gpu_hours, 2),
            'total_estimated_cost': round(total_cost, 2),
            'favorite_experiments': favorite_count,
            'success_rate': round((completed_experiments / max(total_experiments, 1)) * 100, 1)
        }
    
    # Simulation (for demo purposes)
    def _start_metric_simulation(self, experiment_id: int):
        """Simulate metric logging for demo purposes"""
        import threading
        
        def simulate_training():
            epochs = 20
            for epoch in range(epochs):
                # Simulate loss decreasing with some noise
                loss = 4.0 * np.exp(-epoch * 0.2) + random.uniform(-0.1, 0.1)
                val_loss = loss + random.uniform(0.0, 0.3)
                
                # Simulate other metrics
                perplexity = np.exp(loss)
                bleu_score = min(0.8, 0.1 + epoch * 0.035 + random.uniform(-0.05, 0.05))
                rouge_1 = min(0.9, 0.2 + epoch * 0.03 + random.uniform(-0.03, 0.03))
                
                # Log metrics
                self.log_metric(experiment_id, 'loss', loss, 'training', epoch)
                self.log_metric(experiment_id, 'val_loss', val_loss, 'validation', epoch)
                self.log_metric(experiment_id, 'perplexity', perplexity, 'training', epoch)
                self.log_metric(experiment_id, 'bleu_score', bleu_score, 'validation', epoch)
                self.log_metric(experiment_id, 'rouge_1', rouge_1, 'validation', epoch)
                
                time.sleep(0.1)  # Simulate training time
            
            # Complete the experiment
            final_metrics = {
                'final_loss': loss,
                'final_bleu': bleu_score,
                'final_rouge': rouge_1
            }
            self.complete_experiment(experiment_id, final_metrics)
        
        # Start simulation in background
        thread = threading.Thread(target=simulate_training)
        thread.daemon = True
        thread.start()
    
    def initialize_sample_experiments(self):
        """Initialize sample experiments for demo purposes"""
        try:
            # Check if experiments already exist
            if Experiment.query.first():
                return
            
            # Get a model and dataset
            model = LLMModel.query.first()
            dataset = CodingDataset.query.first()
            
            if not model:
                logger.warning("No models found for sample experiments")
                return
            
            sample_experiments = [
                {
                    'name': 'GPT-2 Fine-tuning Baseline',
                    'description': 'Baseline fine-tuning experiment with standard hyperparameters',
                    'hyperparameters': {
                        'learning_rate': 0.0001,
                        'batch_size': 8,
                        'epochs': 20,
                        'lora_r': 8,
                        'lora_alpha': 32
                    },
                    'tags': ['baseline', 'gpt2', 'lora']
                },
                {
                    'name': 'High Learning Rate Experiment',
                    'description': 'Testing higher learning rate for faster convergence',
                    'hyperparameters': {
                        'learning_rate': 0.0005,
                        'batch_size': 8,
                        'epochs': 20,
                        'lora_r': 8,
                        'lora_alpha': 32
                    },
                    'tags': ['high-lr', 'gpt2', 'lora']
                },
                {
                    'name': 'Large LoRA Rank Experiment',
                    'description': 'Testing larger LoRA rank for better adaptation',
                    'hyperparameters': {
                        'learning_rate': 0.0001,
                        'batch_size': 8,
                        'epochs': 20,
                        'lora_r': 16,
                        'lora_alpha': 64
                    },
                    'tags': ['large-lora', 'gpt2', 'lora']
                }
            ]
            
            for exp_data in sample_experiments:
                experiment = self.create_experiment(
                    name=exp_data['name'],
                    description=exp_data['description'],
                    model_id=model.id,
                    hyperparameters=exp_data['hyperparameters'],
                    dataset_id=dataset.id if dataset else None,
                    tags=exp_data['tags'],
                    experiment_group='sample_experiments'
                )
                
                # Start and simulate the experiment
                self.start_experiment(experiment.id)
            
            logger.info("Sample experiments initialized")
            
        except Exception as e:
            logger.error(f"Error initializing sample experiments: {e}")

# Global service instance
experiment_tracking_service = ExperimentTrackingService()