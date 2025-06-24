import threading
import time
import random
import logging
import json
from datetime import datetime
from app import db
from models import TrainingJob, TrainingStatus, LLMModel, ModelStatus

class TrainingService:
    def __init__(self):
        self.active_jobs = {}
        logging.info("Training Service initialized")
    
    def start_training_simulation(self, job_id):
        """Start a simulated training job"""
        job = TrainingJob.query.get(job_id)
        if not job:
            raise ValueError(f"Training job {job_id} not found")
        
        # Update job status
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.utcnow()
        db.session.commit()
        
        # Update model status
        model = LLMModel.query.get(job.model_id)
        model.status = ModelStatus.TRAINING
        db.session.commit()
        
        # Start simulation in background thread
        thread = threading.Thread(target=self._simulate_training, args=(job_id,))
        thread.daemon = True
        thread.start()
        
        self.active_jobs[job_id] = thread
        logging.info(f"Started training simulation for job {job_id}")
    
    def _simulate_training(self, job_id):
        """Simulate the training process"""
        try:
            job = TrainingJob.query.get(job_id)
            total_steps = job.epochs * 100  # Simulate 100 steps per epoch
            
            logs = ["Training started...\n"]
            
            for step in range(total_steps):
                # Simulate training step
                time.sleep(0.1)  # Small delay to simulate processing
                
                current_epoch = (step // 100) + 1
                step_in_epoch = (step % 100) + 1
                progress = (step + 1) / total_steps * 100
                
                # Simulate decreasing loss with some noise
                base_loss = 3.0 - (step / total_steps) * 2.0
                noise = random.uniform(-0.1, 0.1)
                current_loss = max(0.1, base_loss + noise)
                
                # Update job in database
                job = TrainingJob.query.get(job_id)
                job.progress = progress
                job.current_epoch = current_epoch
                job.current_loss = current_loss
                
                # Add log entry every 10 steps
                if step % 10 == 0:
                    log_entry = f"Epoch {current_epoch}/Step {step_in_epoch}: Loss = {current_loss:.4f}\n"
                    logs.append(log_entry)
                    job.logs = "".join(logs[-50:])  # Keep last 50 log entries
                
                db.session.commit()
                
                # Check if job was cancelled
                updated_job = TrainingJob.query.get(job_id)
                if updated_job.status == TrainingStatus.PAUSED:
                    logs.append("Training paused by user.\n")
                    job.logs = "".join(logs[-50:])
                    db.session.commit()
                    return
            
            # Training completed
            job = TrainingJob.query.get(job_id)
            job.status = TrainingStatus.COMPLETED
            job.progress = 100.0
            job.completed_at = datetime.utcnow()
            logs.append("Training completed successfully!\n")
            job.logs = "".join(logs[-50:])
            
            # Update model status
            model = LLMModel.query.get(job.model_id)
            model.status = ModelStatus.AVAILABLE
            model.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            # Remove from active jobs
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            logging.info(f"Training simulation completed for job {job_id}")
            
        except Exception as e:
            logging.error(f"Training simulation failed for job {job_id}: {str(e)}")
            
            # Mark job as failed
            job = TrainingJob.query.get(job_id)
            job.status = TrainingStatus.FAILED
            job.logs = job.logs + f"\nTraining failed: {str(e)}"
            
            # Update model status
            model = LLMModel.query.get(job.model_id)
            model.status = ModelStatus.ERROR
            
            db.session.commit()
            
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
    
    def pause_training(self, job_id):
        """Pause a running training job"""
        job = TrainingJob.query.get(job_id)
        if job and job.status == TrainingStatus.RUNNING:
            job.status = TrainingStatus.PAUSED
            db.session.commit()
            logging.info(f"Training job {job_id} paused")
    
    def resume_training(self, job_id):
        """Resume a paused training job"""
        job = TrainingJob.query.get(job_id)
        if job and job.status == TrainingStatus.PAUSED:
            job.status = TrainingStatus.RUNNING
            db.session.commit()
            
            # Restart simulation thread
            thread = threading.Thread(target=self._simulate_training, args=(job_id,))
            thread.daemon = True
            thread.start()
            self.active_jobs[job_id] = thread
            
            logging.info(f"Training job {job_id} resumed")
    
    def run_evaluation(self, model_id):
        """Run mock evaluation and return metrics"""
        # Simulate evaluation time
        time.sleep(2)
        
        # Generate realistic mock metrics
        metrics = {
            'perplexity': round(random.uniform(15.0, 45.0), 2),
            'bleu_score': round(random.uniform(0.15, 0.85), 3),
            'rouge_score': round(random.uniform(0.20, 0.80), 3),
            'response_diversity': round(random.uniform(0.60, 0.95), 3),
            'avg_response_length': round(random.uniform(50, 150), 1)
        }
        
        logging.info(f"Evaluation completed for model {model_id}: {metrics}")
        return metrics
    
    def simulate_export(self, model_id):
        """Simulate ONNX export process"""
        logging.info(f"Starting ONNX export simulation for model {model_id}")
        
        # Simulate export steps
        steps = [
            "Initializing export...",
            "Loading model weights...",
            "Converting to ONNX format...",
            "Optimizing graph...",
            "Validating exported model...",
            "Export completed!"
        ]
        
        for step in steps:
            time.sleep(1)
            logging.info(f"Export step: {step}")
        
        logging.info(f"ONNX export simulation completed for model {model_id}")
    
    def get_training_statistics(self):
        """Get overall training statistics"""
        total_jobs = TrainingJob.query.count()
        completed_jobs = TrainingJob.query.filter_by(status=TrainingStatus.COMPLETED).count()
        running_jobs = TrainingJob.query.filter_by(status=TrainingStatus.RUNNING).count()
        failed_jobs = TrainingJob.query.filter_by(status=TrainingStatus.FAILED).count()
        
        return {
            'total_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'running_jobs': running_jobs,
            'failed_jobs': failed_jobs,
            'success_rate': round((completed_jobs / total_jobs * 100) if total_jobs > 0 else 0, 1)
        }
