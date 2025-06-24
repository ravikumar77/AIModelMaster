#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) trainer for RLHF
Aligns language models with human preferences using reinforcement learning
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from trl.core import LengthSampler
    import numpy as np
    TRL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TRL library not available: {e}")
    TRL_AVAILABLE = False

from reward_model import RewardTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RLHFTrainer:
    """RLHF trainer using PPO algorithm"""
    
    def __init__(self, 
                 model_name: str,
                 reward_model_path: str,
                 config: Optional[Dict] = None):
        
        self.model_name = model_name
        self.reward_model_path = reward_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default PPO config
        default_config = {
            "learning_rate": 1.41e-5,
            "batch_size": 16,
            "mini_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "optimize_cuda_cache": True,
            "early_stopping": False,
            "target_kl": 6.0,
            "ppo_epochs": 4,
            "max_grad_norm": 0.5,
            "vf_coef": 0.1,
            "cliprange": 0.2,
            "cliprange_value": 0.2,
            "gamma": 1.0,
            "lam": 0.95,
        }
        
        if config:
            default_config.update(config)
        
        self.config = PPOConfig(**default_config) if TRL_AVAILABLE else default_config
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.reward_model = None
        self.ppo_trainer = None
        
    def setup_models(self):
        """Initialize all required models"""
        if not TRL_AVAILABLE:
            logger.error("TRL library not available")
            return False
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load policy model with value head
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model_name)
            
            # Load reference model (frozen copy of original model)
            self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.ref_model.eval()
            
            # Load reward model
            self.reward_model = RewardTrainer()
            if not self.reward_model.load_trained_model(self.reward_model_path):
                logger.error("Failed to load reward model")
                return False
            
            logger.info("All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model setup failed: {e}")
            return False
    
    def create_ppo_trainer(self):
        """Create PPO trainer instance"""
        if not TRL_AVAILABLE:
            return False
        
        try:
            self.ppo_trainer = PPOTrainer(
                config=self.config,
                model=self.model,
                ref_model=self.ref_model,
                tokenizer=self.tokenizer,
            )
            
            logger.info("PPO trainer created successfully")
            return True
            
        except Exception as e:
            logger.error(f"PPO trainer creation failed: {e}")
            return False
    
    def load_prompts(self, prompts_path: str) -> List[str]:
        """Load training prompts"""
        if not os.path.exists(prompts_path):
            logger.warning(f"Prompts file {prompts_path} not found, using sample prompts")
            return self.get_sample_prompts()
        
        try:
            with open(prompts_path, 'r') as f:
                if prompts_path.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, list):
                        prompts = [item if isinstance(item, str) else item.get('prompt', str(item)) for item in data]
                    else:
                        prompts = [str(data)]
                else:
                    prompts = [line.strip() for line in f.readlines() if line.strip()]
            
            logger.info(f"Loaded {len(prompts)} prompts")
            return prompts
            
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            return self.get_sample_prompts()
    
    def get_sample_prompts(self) -> List[str]:
        """Get sample prompts for training"""
        return [
            "Explain the concept of machine learning",
            "Write a professional email requesting a meeting",
            "Describe the process of photosynthesis",
            "What are the benefits of renewable energy?",
            "How does the human brain process information?",
            "Write a short story about artificial intelligence",
            "Explain quantum computing in simple terms",
            "Describe the water cycle",
            "What is the importance of biodiversity?",
            "How do vaccines work?"
        ]
    
    def generate_responses(self, prompts: List[str], max_length: int = 100) -> List[str]:
        """Generate responses using the current policy"""
        if not TRL_AVAILABLE:
            logger.warning("TRL not available, returning mock responses")
            return [f"Mock response to: {prompt[:30]}..." for prompt in prompts]
        
        try:
            responses = []
            
            for prompt in prompts:
                # Tokenize prompt
                inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=len(inputs[0]) + max_length,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode response
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = generated_text[len(prompt):].strip()
                responses.append(response)
            
            return responses
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return []
    
    def compute_rewards(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Compute rewards for generated responses"""
        if not self.reward_model:
            logger.warning("Reward model not available, using mock rewards")
            return [np.random.uniform(0.1, 1.0) for _ in responses]
        
        try:
            rewards = []
            
            for prompt, response in zip(prompts, responses):
                reward = self.reward_model.score_text(prompt, response)
                rewards.append(reward)
            
            # Normalize rewards
            rewards = np.array(rewards)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            return rewards.tolist()
            
        except Exception as e:
            logger.error(f"Reward computation failed: {e}")
            return [0.0] * len(responses)
    
    def train_with_ppo(self,
                      prompts_path: str,
                      output_dir: str = "models/rlhf_model",
                      num_epochs: int = 10,
                      batch_size: int = 16):
        """Train model using PPO algorithm"""
        
        if not TRL_AVAILABLE:
            logger.error("TRL library not available")
            return False
        
        logger.info("Starting RLHF training with PPO")
        
        try:
            # Setup models
            if not self.setup_models():
                return False
            
            if not self.create_ppo_trainer():
                return False
            
            # Load prompts
            prompts = self.load_prompts(prompts_path)
            if not prompts:
                return False
            
            # Training loop
            for epoch in range(num_epochs):
                logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
                
                # Sample batch of prompts
                batch_prompts = np.random.choice(prompts, size=batch_size, replace=True).tolist()
                
                # Tokenize prompts
                prompt_tensors = []
                for prompt in batch_prompts:
                    prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt").squeeze()
                    prompt_tensors.append(prompt_tokens)
                
                # Generate responses
                response_tensors = []
                for prompt_tensor in prompt_tensors:
                    # Generate using PPO model
                    response = self.ppo_trainer.generate(
                        prompt_tensor.unsqueeze(0),
                        max_length=150,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True
                    )
                    response_tensors.append(response.squeeze())
                
                # Decode responses
                responses = []
                for i, response_tensor in enumerate(response_tensors):
                    full_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
                    prompt_text = batch_prompts[i]
                    response_text = full_text[len(prompt_text):].strip()
                    responses.append(response_text)
                
                # Compute rewards
                rewards = self.compute_rewards(batch_prompts, responses)
                reward_tensors = [torch.tensor(reward) for reward in rewards]
                
                # PPO step
                stats = self.ppo_trainer.step(prompt_tensors, response_tensors, reward_tensors)
                
                # Log statistics
                if epoch % 5 == 0:
                    logger.info(f"Epoch {epoch + 1} stats: {stats}")
                    
                    # Log sample generation
                    sample_prompt = batch_prompts[0]
                    sample_response = responses[0]
                    sample_reward = rewards[0]
                    
                    logger.info(f"Sample - Prompt: {sample_prompt}")
                    logger.info(f"Sample - Response: {sample_response}")
                    logger.info(f"Sample - Reward: {sample_reward:.3f}")
            
            # Save trained model
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Save training config
            training_config = {
                "base_model": self.model_name,
                "reward_model_path": self.reward_model_path,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "ppo_config": self.config.to_dict(),
                "training_date": datetime.now().isoformat()
            }
            
            with open(os.path.join(output_dir, "rlhf_config.json"), 'w') as f:
                json.dump(training_config, f, indent=2)
            
            logger.info(f"RLHF model saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"RLHF training failed: {e}")
            return False
    
    def evaluate_alignment(self, test_prompts: List[str]) -> Dict[str, float]:
        """Evaluate model alignment using test prompts"""
        try:
            # Generate responses
            responses = self.generate_responses(test_prompts)
            
            # Compute rewards
            rewards = self.compute_rewards(test_prompts, responses)
            
            # Calculate metrics
            metrics = {
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "min_reward": np.min(rewards),
                "max_reward": np.max(rewards),
                "num_positive_rewards": sum(1 for r in rewards if r > 0),
                "positive_reward_ratio": sum(1 for r in rewards if r > 0) / len(rewards)
            }
            
            logger.info(f"Alignment evaluation: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Alignment evaluation failed: {e}")
            return {}

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Train model with RLHF using PPO")
    parser.add_argument("--model", required=True, help="Base model name")
    parser.add_argument("--reward-model", required=True, help="Path to trained reward model")
    parser.add_argument("--prompts", required=True, help="Path to training prompts")
    parser.add_argument("--output", default="models/rlhf_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1.41e-5, help="Learning rate")
    parser.add_argument("--evaluate", help="Path to test prompts for evaluation")
    
    args = parser.parse_args()
    
    # Create trainer
    config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size
    }
    
    trainer = RLHFTrainer(
        model_name=args.model,
        reward_model_path=args.reward_model,
        config=config
    )
    
    # Train model
    success = trainer.train_with_ppo(
        prompts_path=args.prompts,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate if test prompts provided
    if success and args.evaluate:
        test_prompts = trainer.load_prompts(args.evaluate)
        metrics = trainer.evaluate_alignment(test_prompts)
        print(f"Alignment metrics: {metrics}")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()