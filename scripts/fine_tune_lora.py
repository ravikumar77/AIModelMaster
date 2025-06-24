#!/usr/bin/env python3
"""
LoRA/QLoRA fine-tuning script for efficient model adaptation
Uses PEFT library for parameter-efficient fine-tuning
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
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
    from peft import (
        get_peft_model, LoraConfig, TaskType,
        PeftModel, PeftConfig
    )
    from datasets import Dataset
    import bitsandbytes as bnb
    PEFT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PEFT libraries not available: {e}")
    PEFT_AVAILABLE = False

from app import app, db
from models import LLMModel, TrainingJob, TrainingStatus, ModelStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoRATrainer:
    def __init__(self, config_path="config/model_config.yaml"):
        """Initialize LoRA trainer"""
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def load_config(self, config_path):
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self.get_default_config()
    
    def get_default_config(self):
        """Default configuration"""
        return {
            'lora': {
                'default_r': 8,
                'default_alpha': 32,
                'default_dropout': 0.05
            },
            'training': {
                'default_epochs': 3,
                'default_learning_rate': 0.0001,
                'default_batch_size': 8
            }
        }
    
    def create_lora_config(self, r=None, alpha=None, dropout=None, target_modules=None):
        """Create LoRA configuration"""
        if not PEFT_AVAILABLE:
            logger.error("PEFT library not available")
            return None
            
        r = r or self.config['lora']['default_r']
        alpha = alpha or self.config['lora']['default_alpha']
        dropout = dropout or self.config['lora']['default_dropout']
        
        # Default target modules for causal LM
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]
        
        return LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
    
    def prepare_model_for_training(self, model_name, lora_config, load_in_8bit=False):
        """Prepare model with LoRA adapters"""
        if not PEFT_AVAILABLE:
            logger.error("PEFT library not available")
            return None, None
            
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        # Add LoRA adapters
        model = get_peft_model(model, lora_config)
        
        logger.info(f"LoRA model created with {model.num_parameters()} parameters")
        logger.info(f"Trainable parameters: {model.get_nb_trainable_parameters()}")
        
        return model, tokenizer
    
    def prepare_dataset(self, data_path, tokenizer, max_length=512):
        """Prepare dataset for fine-tuning"""
        if not os.path.exists(data_path):
            logger.error(f"Data path {data_path} does not exist")
            raise FileNotFoundError(f"Data path {data_path} not found")
        
        # Load data
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # Support different JSON formats
            if isinstance(data, list):
                if isinstance(data[0], dict):
                    # List of objects with 'text' field
                    texts = [item.get('text', str(item)) for item in data]
                else:
                    # List of strings
                    texts = [str(item) for item in data]
            else:
                # Single object or other format
                texts = [str(data)]
                
        elif data_path.endswith('.txt'):
            with open(data_path, 'r') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]
        else:
            logger.error(f"Unsupported file format: {data_path}")
            raise ValueError(f"Unsupported file format")
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=max_length
            )
        
        dataset = Dataset.from_dict({'text': texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def fine_tune(self,
                  model_name,
                  data_path,
                  output_dir="models/finetuned",
                  lora_r=None,
                  lora_alpha=None,
                  lora_dropout=None,
                  epochs=None,
                  learning_rate=None,
                  batch_size=None,
                  load_in_8bit=False,
                  job_id=None):
        """Fine-tune model with LoRA"""
        
        if not PEFT_AVAILABLE:
            logger.error("PEFT library not available")
            return False
        
        # Get parameters
        epochs = epochs or self.config['training']['default_epochs']
        learning_rate = learning_rate or self.config['training']['default_learning_rate']
        batch_size = batch_size or self.config['training']['default_batch_size']
        
        logger.info(f"Starting LoRA fine-tuning for {model_name}")
        logger.info(f"LoRA params: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        logger.info(f"Training params: epochs={epochs}, lr={learning_rate}, batch_size={batch_size}")
        
        try:
            # Create LoRA config
            lora_config = self.create_lora_config(lora_r, lora_alpha, lora_dropout)
            
            # Prepare model
            model, tokenizer = self.prepare_model_for_training(
                model_name, lora_config, load_in_8bit
            )
            
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
                dataloader_pin_memory=False,
                remove_unused_columns=False
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                tokenizer=tokenizer
            )
            
            # Update job status
            if job_id:
                with app.app_context():
                    job = TrainingJob.query.get(job_id)
                    if job:
                        job.status = TrainingStatus.RUNNING
                        job.started_at = datetime.utcnow()
                        db.session.commit()
            
            # Train
            logger.info("Starting LoRA fine-tuning...")
            trainer.train()
            
            # Save the fine-tuned model
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            
            # Save LoRA adapters separately
            model.save_pretrained(f"{output_dir}/lora_adapters")
            
            logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
            
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
            logger.error(f"Fine-tuning failed: {e}")
            
            # Update job status on failure
            if job_id:
                with app.app_context():
                    job = TrainingJob.query.get(job_id)
                    if job:
                        job.status = TrainingStatus.FAILED
                        job.logs = f"Fine-tuning failed: {str(e)}"
                        db.session.commit()
                        
                        # Update model status
                        model_record = LLMModel.query.get(job.model_id)
                        if model_record:
                            model_record.status = ModelStatus.ERROR
                            db.session.commit()
            
            return False

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA")
    parser.add_argument("--model", required=True, help="Base model name")
    parser.add_argument("--data", required=True, help="Path to training data")
    parser.add_argument("--output", default="models/finetuned", help="Output directory")
    parser.add_argument("--lora-r", type=int, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, help="LoRA dropout")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument("--job-id", type=int, help="Training job ID")
    
    args = parser.parse_args()
    
    trainer = LoRATrainer()
    success = trainer.fine_tune(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        load_in_8bit=args.load_in_8bit,
        job_id=args.job_id
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()