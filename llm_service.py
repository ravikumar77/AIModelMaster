import os
import logging
import time
from models import LLMModel

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. Using mock mode.")

class LLMService:
    def __init__(self):
        self.loaded_models = {}
        self.tokenizers = {}
        if TRANSFORMERS_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"LLM Service initialized on device: {self.device}")
        else:
            self.device = "cpu"
            logging.info("LLM Service initialized in mock mode (transformers not available)")
    
    def get_available_models(self):
        """Get list of available pre-trained models"""
        return [
            "distilgpt2",
            "gpt2",
            "microsoft/DialoGPT-small",
            "facebook/opt-125m",
            "EleutherAI/gpt-neo-125M"
        ]
    
    def get_model_size(self, model_name):
        """Get estimated model size"""
        size_map = {
            "distilgpt2": "82M",
            "gpt2": "124M",
            "microsoft/DialoGPT-small": "117M",
            "facebook/opt-125m": "125M",
            "EleutherAI/gpt-neo-125M": "125M"
        }
        return size_map.get(model_name, "Unknown")
    
    def load_model(self, model_id):
        """Load a model for inference"""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id], self.tokenizers[model_id]
        
        try:
            model_record = LLMModel.query.get(model_id)
            if not model_record:
                raise ValueError(f"Model with ID {model_id} not found")
            
            model_name = model_record.base_model
            
            if not TRANSFORMERS_AVAILABLE:
                # Mock model loading
                logging.info(f"Mock loading model: {model_name}")
                self.loaded_models[model_id] = f"mock_model_{model_id}"
                self.tokenizers[model_id] = f"mock_tokenizer_{model_id}"
                return self.loaded_models[model_id], self.tokenizers[model_id]
            
            logging.info(f"Loading model: {model_name}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            # Cache the loaded model
            self.loaded_models[model_id] = model
            self.tokenizers[model_id] = tokenizer
            
            logging.info(f"Model {model_name} loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logging.error(f"Error loading model {model_id}: {str(e)}")
            raise
    
    def generate_text(self, model_id, prompt, temperature=0.7, max_length=100, top_p=0.9, top_k=50):
        """Generate text using the specified model"""
        try:
            model, tokenizer = self.load_model(model_id)
            
            if not TRANSFORMERS_AVAILABLE:
                # Mock text generation
                import random
                time.sleep(1)  # Simulate processing time
                
                # Generate mock response based on prompt
                mock_responses = [
                    f"This is a generated response to: '{prompt[:50]}...' with temperature {temperature}",
                    f"Based on your prompt '{prompt[:30]}...', here's a creative continuation with {max_length} max tokens.",
                    f"AI-generated text (mock): {prompt} [Generated with temp={temperature}, top_p={top_p}]",
                    f"Sample output for '{prompt[:40]}...' - this would be actual model output in production."
                ]
                
                base_response = random.choice(mock_responses)
                
                # Add some variation based on temperature
                if temperature > 1.0:
                    base_response += " (High creativity mode enabled)"
                elif temperature < 0.3:
                    base_response += " (Conservative generation mode)"
                    
                return base_response[:max_length]
            
            # Tokenize input
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from the output
            response = generated_text[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logging.error(f"Error generating text: {str(e)}")
            raise Exception(f"Text generation failed: {str(e)}")
    
    def unload_model(self, model_id):
        """Unload a model from memory"""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            del self.tokenizers[model_id]
            if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info(f"Model {model_id} unloaded from memory")
    
    def get_model_info(self, model_id):
        """Get information about a loaded model"""
        model_record = LLMModel.query.get(model_id)
        if not model_record:
            return None
        
        is_loaded = model_id in self.loaded_models
        
        return {
            "id": model_id,
            "name": model_record.name,
            "base_model": model_record.base_model,
            "status": model_record.status.value,
            "is_loaded": is_loaded,
            "model_size": model_record.model_size
        }
