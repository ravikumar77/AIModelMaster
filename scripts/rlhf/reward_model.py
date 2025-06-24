#!/usr/bin/env python3
"""
Reward model training for RLHF (Reinforcement Learning with Human Feedback)
Trains a model to predict human preferences for text generation quality
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, AutoConfig
    )
    from datasets import Dataset as HFDataset
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch/Transformers not available: {e}")
    TORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PreferenceDataset(Dataset):
    """Dataset for preference learning"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format: prompt + " " + response
        chosen_text = item['prompt'] + " " + item['chosen']
        rejected_text = item['prompt'] + " " + item['rejected']
        
        # Tokenize both texts
        chosen_encoding = self.tokenizer(
            chosen_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        rejected_encoding = self.tokenizer(
            rejected_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_encoding['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_encoding['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_encoding['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_encoding['attention_mask'].squeeze(),
        }

class RewardModel(nn.Module):
    """Reward model for scoring text quality"""
    
    def __init__(self, base_model_name, num_labels=1):
        super().__init__()
        
        config = AutoConfig.from_pretrained(base_model_name)
        config.num_labels = num_labels
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, config=config
        )
        
        # Replace classifier head with a single output (reward score)
        if hasattr(self.model, 'classifier'):
            self.model.classifier = nn.Linear(self.model.config.hidden_size, 1)
        elif hasattr(self.model, 'score'):
            self.model.score = nn.Linear(self.model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)  # Remove last dimension to get scalar scores

class RewardTrainer:
    """Trainer for reward models"""
    
    def __init__(self, base_model_name="distilbert-base-uncased"):
        self.base_model_name = base_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
    def load_preference_data(self, data_path: str) -> List[Dict]:
        """Load preference data from file"""
        if not os.path.exists(data_path):
            logger.warning(f"Data file {data_path} not found, creating sample data")
            return self.create_sample_preference_data()
        
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # Validate data format
            required_keys = ['prompt', 'chosen', 'rejected']
            if not all(key in data[0] for key in required_keys):
                logger.error(f"Data must contain keys: {required_keys}")
                return []
            
            logger.info(f"Loaded {len(data)} preference examples")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load preference data: {e}")
            return []
    
    def create_sample_preference_data(self) -> List[Dict]:
        """Create sample preference data for testing"""
        sample_data = [
            {
                "prompt": "Explain artificial intelligence",
                "chosen": "Artificial intelligence (AI) is a branch of computer science that aims to create machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving.",
                "rejected": "AI is just computers doing stuff that humans do but worse and it's probably going to take over the world someday."
            },
            {
                "prompt": "Write a professional email",
                "chosen": "Dear Mr. Smith,\n\nI hope this email finds you well. I am writing to follow up on our previous conversation regarding the quarterly report. Please let me know if you need any additional information.\n\nBest regards,\nJohn Doe",
                "rejected": "hey smith, whats up with that report thing? hit me back when u get this. thx"
            },
            {
                "prompt": "Describe the water cycle",
                "chosen": "The water cycle is the continuous movement of water through Earth's atmosphere, land, and oceans. It involves evaporation, condensation, precipitation, and collection, driven by solar energy and gravity.",
                "rejected": "Water goes up and comes down and repeats forever because of the sun or something like that."
            }
        ]
        
        logger.info(f"Created {len(sample_data)} sample preference examples")
        return sample_data
    
    def prepare_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available")
            return False
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create reward model
            self.model = RewardModel(self.base_model_name)
            self.model.to(self.device)
            
            logger.info(f"Initialized reward model based on {self.base_model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            return False
    
    def train_reward_model(self,
                          data_path: str,
                          output_dir: str = "models/reward_model",
                          epochs: int = 3,
                          learning_rate: float = 5e-5,
                          batch_size: int = 4):
        """Train the reward model on preference data"""
        
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available")
            return False
        
        logger.info("Starting reward model training")
        
        try:
            # Load data
            preference_data = self.load_preference_data(data_path)
            if not preference_data:
                logger.error("No preference data available")
                return False
            
            # Initialize model
            if not self.prepare_model_and_tokenizer():
                return False
            
            # Create dataset
            dataset = PreferenceDataset(preference_data, self.tokenizer)
            
            # Custom training loop for preference learning
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            self.model.train()
            
            for epoch in range(epochs):
                total_loss = 0
                num_batches = 0
                
                for batch in dataloader:
                    # Move to device
                    chosen_input_ids = batch['chosen_input_ids'].to(self.device)
                    chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
                    rejected_input_ids = batch['rejected_input_ids'].to(self.device)
                    rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
                    
                    # Get rewards for chosen and rejected responses
                    chosen_rewards = self.model(chosen_input_ids, chosen_attention_mask)
                    rejected_rewards = self.model(rejected_input_ids, rejected_attention_mask)
                    
                    # Preference loss: chosen should have higher reward than rejected
                    # Using margin ranking loss
                    loss = torch.nn.functional.margin_ranking_loss(
                        chosen_rewards,
                        rejected_rewards,
                        torch.ones_like(chosen_rewards),
                        margin=0.1
                    )
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches
                logger.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Save model
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(output_dir, "reward_model.pt"))
            self.tokenizer.save_pretrained(output_dir)
            
            # Save model config
            config = {
                "base_model": self.base_model_name,
                "training_epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "training_date": datetime.now().isoformat()
            }
            
            with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Reward model saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def load_trained_model(self, model_path: str):
        """Load a trained reward model"""
        try:
            if not self.prepare_model_and_tokenizer():
                return False
            
            state_dict = torch.load(
                os.path.join(model_path, "reward_model.pt"),
                map_location=self.device
            )
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            logger.info(f"Loaded reward model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def score_text(self, prompt: str, response: str) -> float:
        """Score a text response given a prompt"""
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded")
            return 0.0
        
        try:
            text = prompt + " " + response
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                score = self.model(
                    encoding['input_ids'],
                    encoding['attention_mask']
                )
            
            return float(score.item())
            
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return 0.0
    
    def evaluate_model(self, test_data_path: str) -> Dict[str, float]:
        """Evaluate the reward model on test data"""
        try:
            test_data = self.load_preference_data(test_data_path)
            if not test_data:
                return {}
            
            correct_preferences = 0
            total_preferences = len(test_data)
            
            for item in test_data:
                chosen_score = self.score_text(item['prompt'], item['chosen'])
                rejected_score = self.score_text(item['prompt'], item['rejected'])
                
                if chosen_score > rejected_score:
                    correct_preferences += 1
            
            accuracy = correct_preferences / total_preferences
            
            metrics = {
                "preference_accuracy": accuracy,
                "total_examples": total_preferences,
                "correct_preferences": correct_preferences
            }
            
            logger.info(f"Evaluation results: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Train reward model for RLHF")
    parser.add_argument("--data", required=True, help="Path to preference data (JSON)")
    parser.add_argument("--output", default="models/reward_model", help="Output directory")
    parser.add_argument("--base-model", default="distilbert-base-uncased", 
                       help="Base model for reward model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--evaluate", help="Path to test data for evaluation")
    
    args = parser.parse_args()
    
    trainer = RewardTrainer(args.base_model)
    
    # Train model
    success = trainer.train_reward_model(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )
    
    # Evaluate if test data provided
    if success and args.evaluate:
        trainer.load_trained_model(args.output)
        metrics = trainer.evaluate_model(args.evaluate)
        print(f"Evaluation metrics: {metrics}")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()