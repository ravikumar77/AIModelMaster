"""
Custom Training Service - Handle custom model training with user datasets
"""
import os
import json
import logging
import threading
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from app import db
from models import (
    CustomTrainingJob, CustomDataset, LLMModel, TrainingStatus, 
    ModelStatus, TrainingCheckpoint, User
)

class CustomTrainingService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.training_threads = {}
        self.available_base_models = [
            {'name': 'distilgpt2', 'size': '82M', 'description': 'Fast and efficient GPT-2 variant'},
            {'name': 'gpt2', 'size': '124M', 'description': 'Original GPT-2 small model'},
            {'name': 'gpt2-medium', 'size': '355M', 'description': 'Medium-sized GPT-2 model'},
            {'name': 'microsoft/DialoGPT-small', 'size': '117M', 'description': 'Conversational AI model'},
            {'name': 'facebook/opt-125m', 'size': '125M', 'description': 'Open Pretrained Transformer'},
            {'name': 'EleutherAI/gpt-neo-125M', 'size': '125M', 'description': 'GPT-Neo model'}
        ]
        
        # Ensure directories exist
        os.makedirs('models/custom', exist_ok=True)
        os.makedirs('jobs/training_runs', exist_ok=True)
    
    def get_available_base_models(self) -> List[Dict]:
        """Get list of available base models for training"""
        return self.available_base_models
    
    def create_training_job(self, job_data: Dict, user_id: int) -> Dict:
        """Create a new custom training job"""
        try:
            # Validate dataset exists and belongs to user
            dataset = CustomDataset.query.filter_by(
                id=job_data['dataset_id'], 
                created_by=user_id
            ).first()
            
            if not dataset:
                return {"success": False, "error": "Dataset not found or access denied"}
            
            if not dataset.is_processed:
                return {"success": False, "error": "Dataset is not processed yet"}
            
            # Create training job
            job = CustomTrainingJob(
                job_name=job_data['job_name'],
                base_model=job_data['base_model'],
                dataset_id=job_data['dataset_id'],
                epochs=job_data.get('epochs', 3),
                learning_rate=job_data.get('learning_rate', 0.0001),
                batch_size=job_data.get('batch_size', 8),
                max_length=job_data.get('max_length', 512),
                warmup_steps=job_data.get('warmup_steps', 500),
                use_lora=job_data.get('use_lora', True),
                lora_r=job_data.get('lora_r', 8),
                lora_alpha=job_data.get('lora_alpha', 32),
                lora_dropout=job_data.get('lora_dropout', 0.05),
                use_qlora=job_data.get('use_qlora', False),
                save_checkpoints=job_data.get('save_checkpoints', True),
                checkpoint_frequency=job_data.get('checkpoint_frequency', 500),
                output_model_name=job_data.get('output_model_name', f"{job_data['job_name']}_model"),
                created_by=user_id
            )
            
            db.session.add(job)
            db.session.commit()
            
            return {
                "success": True,
                "job_id": job.id,
                "message": "Training job created successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Error creating training job: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def start_training_job(self, job_id: int, user_id: int) -> Dict:
        """Start a training job"""
        try:
            job = CustomTrainingJob.query.filter_by(
                id=job_id, 
                created_by=user_id
            ).first()
            
            if not job:
                return {"success": False, "error": "Training job not found"}
            
            if job.status != TrainingStatus.PENDING:
                return {"success": False, "error": f"Job is in {job.status.value} state"}
            
            # Update job status
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.utcnow()
            db.session.commit()
            
            # Start training in background thread
            thread = threading.Thread(
                target=self._simulate_training,
                args=(job_id,),
                daemon=True
            )
            thread.start()
            self.training_threads[job_id] = thread
            
            return {
                "success": True,
                "message": "Training job started successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Error starting training job: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _simulate_training(self, job_id: int):
        """Simulate training process with realistic progress"""
        try:
            job = CustomTrainingJob.query.get(job_id)
            if not job:
                return
            
            self.logger.info(f"Starting training simulation for job {job_id}")
            
            # Training simulation parameters
            total_steps = job.epochs * 100  # Simulate 100 steps per epoch
            current_step = 0
            
            # Initialize metrics
            initial_loss = 4.0 + random.uniform(-0.5, 0.5)
            current_loss = initial_loss
            metrics_history = []
            
            for epoch in range(job.epochs):
                job.current_epoch = epoch + 1
                epoch_start_loss = current_loss
                
                # Simulate steps within epoch
                for step in range(100):
                    current_step += 1
                    
                    # Simulate loss decrease with some noise
                    loss_reduction = (initial_loss - 1.5) * (current_step / total_steps)
                    noise = random.uniform(-0.1, 0.1)
                    current_loss = initial_loss - loss_reduction + noise
                    current_loss = max(current_loss, 0.8)  # Minimum loss
                    
                    # Update job progress
                    job.progress = (current_step / total_steps) * 100
                    job.current_loss = current_loss
                    
                    if current_loss < (job.best_loss or float('inf')):
                        job.best_loss = current_loss
                    
                    # Add to metrics history
                    metrics_history.append({
                        'step': current_step,
                        'epoch': epoch + 1,
                        'loss': current_loss,
                        'learning_rate': job.learning_rate,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
                    # Update training logs
                    if current_step % 20 == 0:  # Log every 20 steps
                        log_entry = f"Step {current_step}/{total_steps} | Epoch {epoch+1}/{job.epochs} | Loss: {current_loss:.4f}"
                        if job.training_logs:
                            job.training_logs += f"\n{log_entry}"
                        else:
                            job.training_logs = log_entry
                    
                    # Save checkpoint periodically
                    if job.save_checkpoints and current_step % job.checkpoint_frequency == 0:
                        self._create_checkpoint(job, epoch + 1, current_step, current_loss)
                    
                    # Simulate training time
                    time.sleep(0.5)  # Half second per step for demo
                    
                    # Check if job was cancelled
                    db.session.refresh(job)
                    if job.status != TrainingStatus.RUNNING:
                        self.logger.info(f"Training job {job_id} was cancelled")
                        return
                    
                    # Update database
                    db.session.commit()
                
                # End of epoch logging
                epoch_end_log = f"Epoch {epoch+1} completed | Loss: {current_loss:.4f} | Progress: {job.progress:.1f}%"
                job.training_logs += f"\n{epoch_end_log}"
                db.session.commit()
            
            # Training completed
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress = 100.0
            job.metrics_data = json.dumps(metrics_history)
            
            # Create final model
            result_model = self._create_result_model(job)
            if result_model:
                job.model_id = result_model.id
            
            # Final checkpoint
            if job.save_checkpoints:
                self._create_checkpoint(job, job.epochs, total_steps, current_loss, is_final=True)
            
            job.training_logs += f"\nTraining completed successfully! Final loss: {current_loss:.4f}"
            db.session.commit()
            
            self.logger.info(f"Training job {job_id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in training simulation: {str(e)}")
            job = CustomTrainingJob.query.get(job_id)
            if job:
                job.status = TrainingStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                db.session.commit()
    
    def _create_checkpoint(self, job: CustomTrainingJob, epoch: int, step: int, 
                          loss: float, is_final: bool = False):
        """Create a training checkpoint"""
        try:
            checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}"
            if is_final:
                checkpoint_name = f"final_model_{job.job_name}"
            
            # Simulate checkpoint file
            checkpoint_dir = f"jobs/training_runs/{job.id}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")
            
            # Create dummy checkpoint file
            with open(checkpoint_path, 'w') as f:
                f.write(f"Checkpoint for job {job.id} at epoch {epoch}, step {step}")
            
            # Create database record
            checkpoint = TrainingCheckpoint(
                training_job_id=job.id,
                checkpoint_name=checkpoint_name,
                file_path=checkpoint_path,
                epoch=epoch,
                step=step,
                loss_value=loss,
                file_size=1024  # Dummy size
            )
            
            db.session.add(checkpoint)
            db.session.commit()
            
        except Exception as e:
            self.logger.error(f"Error creating checkpoint: {str(e)}")
    
    def _create_result_model(self, job: CustomTrainingJob) -> Optional[LLMModel]:
        """Create a model record for the training result"""
        try:
            model = LLMModel(
                name=job.output_model_name,
                base_model=job.base_model,
                status=ModelStatus.AVAILABLE,
                description=f"Custom trained model from job: {job.job_name}",
                model_size="Custom",
                parameters=json.dumps({
                    'training_job_id': job.id,
                    'base_model': job.base_model,
                    'dataset_samples': job.dataset.num_samples if job.dataset else 0,
                    'epochs': job.epochs,
                    'learning_rate': job.learning_rate,
                    'lora_config': {
                        'r': job.lora_r,
                        'alpha': job.lora_alpha,
                        'dropout': job.lora_dropout
                    } if job.use_lora else None,
                    'final_loss': job.best_loss
                })
            )
            
            db.session.add(model)
            db.session.commit()
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating result model: {str(e)}")
            return None
    
    def get_training_jobs(self, user_id: int) -> List[CustomTrainingJob]:
        """Get all training jobs for a user"""
        return CustomTrainingJob.query.filter_by(
            created_by=user_id
        ).order_by(CustomTrainingJob.created_at.desc()).all()
    
    def get_training_job(self, job_id: int, user_id: int = None) -> Optional[CustomTrainingJob]:
        """Get a specific training job"""
        query = CustomTrainingJob.query.filter_by(id=job_id)
        if user_id:
            query = query.filter_by(created_by=user_id)
        return query.first()
    
    def stop_training_job(self, job_id: int, user_id: int) -> Dict:
        """Stop a running training job"""
        try:
            job = self.get_training_job(job_id, user_id)
            if not job:
                return {"success": False, "error": "Training job not found"}
            
            if job.status != TrainingStatus.RUNNING:
                return {"success": False, "error": "Job is not running"}
            
            job.status = TrainingStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error_message = "Stopped by user"
            db.session.commit()
            
            return {"success": True, "message": "Training job stopped"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_job_metrics(self, job_id: int, user_id: int) -> Dict:
        """Get training metrics for visualization"""
        try:
            job = self.get_training_job(job_id, user_id)
            if not job:
                return {"success": False, "error": "Training job not found"}
            
            metrics = []
            if job.metrics_data:
                metrics = json.loads(job.metrics_data)
            
            return {
                "success": True,
                "metrics": metrics,
                "job_info": {
                    "name": job.job_name,
                    "status": job.status.value,
                    "progress": job.progress,
                    "current_loss": job.current_loss,
                    "best_loss": job.best_loss,
                    "current_epoch": job.current_epoch,
                    "total_epochs": job.epochs
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_job_checkpoints(self, job_id: int, user_id: int) -> List[TrainingCheckpoint]:
        """Get checkpoints for a training job"""
        job = self.get_training_job(job_id, user_id)
        if not job:
            return []
        
        return TrainingCheckpoint.query.filter_by(
            training_job_id=job_id
        ).order_by(TrainingCheckpoint.step.desc()).all()
    
    def delete_training_job(self, job_id: int, user_id: int) -> Dict:
        """Delete a training job and its associated files"""
        try:
            job = self.get_training_job(job_id, user_id)
            if not job:
                return {"success": False, "error": "Training job not found"}
            
            if job.status == TrainingStatus.RUNNING:
                return {"success": False, "error": "Cannot delete running job"}
            
            # Remove checkpoints
            checkpoints = self.get_job_checkpoints(job_id, user_id)
            for checkpoint in checkpoints:
                if os.path.exists(checkpoint.file_path):
                    os.remove(checkpoint.file_path)
                db.session.delete(checkpoint)
            
            # Remove job directory
            job_dir = f"jobs/training_runs/{job_id}"
            if os.path.exists(job_dir):
                import shutil
                shutil.rmtree(job_dir)
            
            # Remove job
            db.session.delete(job)
            db.session.commit()
            
            return {"success": True, "message": "Training job deleted successfully"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

# Global service instance
custom_training_service = CustomTrainingService()