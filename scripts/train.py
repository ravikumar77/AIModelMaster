#!/usr/bin/env python3
"""
Full training script for LLM from scratch using PyTorch and Hugging Face
Supports training on custom datasets with configurable parameters
"""

import os
import sys
import json
import yaml
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
    from datasets import Dataset, load_dataset
    import wandb
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: {e}")
    TRANSFORMERS_AVAILABLE = False

from app import app, db
from models import LLMModel, TrainingJob, TrainingStatus, ModelStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLMTrainer:
    def __init__(self, config_path="config/model_config.yaml"):
        """Initialize the LLM trainer with configuration"""
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration"""
        return {
            'training': {
                'default_epochs': 3,
                'default_learning_rate': 0.0001,
                'default_batch_size': 8
            },
            'base_models': {
                'gpt2': {'size': '124M'},
                'distilgpt2': {'size': '82M'}
            }
        }
    
    def prepare_dataset(self, data_path, tokenizer, max_length=512):
        """Prepare dataset for training"""
        if not os.path.exists(data_path):
            logger.error(f"Data path {data_path} does not exist")
            raise FileNotFoundError(f"Data path {data_path} not found")
        
        # Load data from file
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
            texts = [item['text'] if isinstance(item, dict) else str(item) for item in data]
        elif data_path.endswith('.txt'):
            with open(data_path, 'r') as f:
                texts = f.readlines()
        else:
            # Try to load as HuggingFace dataset
            try:
                dataset = load_dataset(data_path)
                texts = dataset['train']['text']
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                raise
        
        # Tokenize texts
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=max_length,
                return_overflowing_tokens=True
            )
        
        # Create dataset
        dataset = Dataset.from_dict({'text': texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train_model(self, 
                   model_name, 
                   data_path, 
                   output_dir="models/trained",
                   epochs=None,
                   learning_rate=None,
                   batch_size=None,
                   job_id=None):
        """Train a model from scratch"""
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available")
            return False
        
        # Get training parameters
        epochs = epochs or self.config['training']['default_epochs']
        learning_rate = learning_rate or self.config['training']['default_learning_rate']
        batch_size = batch_size or self.config['training']['default_batch_size']
        
        logger.info(f"Starting training for {model_name}")
        logger.info(f"Parameters: epochs={epochs}, lr={learning_rate}, batch_size={batch_size}")
        
        try:
            # Initialize wandb if available
            try:
                wandb.init(
                    project="llm-training",
                    name=f"train-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    config={
                        'model_name': model_name,
                        'epochs': epochs,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.to(self.device)
            
            # Prepare dataset
            train_dataset = self.prepare_dataset(data_path, tokenizer)
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                save_steps=500,
                save_total_limit=2,
                prediction_loss_only=True,
                learning_rate=learning_rate,
                warmup_steps=100,
                logging_steps=50,
                logging_dir=f"{output_dir}/logs",
                report_to="wandb" if 'wandb' in globals() else None,
                dataloader_pin_memory=False
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                tokenizer=tokenizer
            )
            
            # Update job status if provided
            if job_id:
                with app.app_context():
                    job = TrainingJob.query.get(job_id)
                    if job:
                        job.status = TrainingStatus.RUNNING
                        job.started_at = datetime.utcnow()
                        db.session.commit()
            
            # Train the model
            logger.info("Starting training...")
            trainer.train()
            
            # Save the model
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Training completed. Model saved to {output_dir}")
            
            # Update job status
            if job_id:
                with app.app_context():
                    job = TrainingJob.query.get(job_id)
                    if job:
                        job.status = TrainingStatus.COMPLETED
                        job.completed_at = datetime.utcnow()
                        job.progress = 100.0
                        db.session.commit()
                        
                        # Update model status
                        model_record = LLMModel.query.get(job.model_id)
                        if model_record:
                            model_record.status = ModelStatus.AVAILABLE
                            db.session.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            
            # Update job status on failure
            if job_id:
                with app.app_context():
                    job = TrainingJob.query.get(job_id)
                    if job:
                        job.status = TrainingStatus.FAILED
                        job.logs = f"Training failed: {str(e)}"
                        db.session.commit()
                        
                        # Update model status
                        model_record = LLMModel.query.get(job.model_id)
                        if model_record:
                            model_record.status = ModelStatus.ERROR
                            db.session.commit()
            
            return False
        
        finally:
            try:
                wandb.finish()
            except:
                pass

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Train LLM from scratch")
    parser.add_argument("--model", required=True, help="Base model name")
    parser.add_argument("--data", required=True, help="Path to training data")
    parser.add_argument("--output", default="models/trained", help="Output directory")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--job-id", type=int, help="Training job ID")
    
    args = parser.parse_args()
    
    trainer = LLMTrainer()
    success = trainer.train_model(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        job_id=args.job_id
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()