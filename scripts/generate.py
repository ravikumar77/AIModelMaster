#!/usr/bin/env python3
"""
Text generation script for inference using trained models
Supports both PyTorch and ONNX models with streaming output
"""

import os
import sys
import json
import yaml
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
    import onnxruntime as ort
    import numpy as np
    INFERENCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Inference libraries not available: {e}")
    INFERENCE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextGenerator:
    def __init__(self, config_path="config/model_config.yaml"):
        """Initialize text generator"""
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.onnx_session = None
        
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
            'api': {
                'generation': {
                    'default_temperature': 0.7,
                    'default_max_length': 100,
                    'default_top_p': 0.9,
                    'default_top_k': 50
                }
            }
        }
    
    def load_pytorch_model(self, model_path):
        """Load PyTorch model for inference"""
        if not INFERENCE_AVAILABLE:
            logger.error("Inference libraries not available")
            return False
            
        try:
            logger.info(f"Loading PyTorch model from {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("PyTorch model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            return False
    
    def load_onnx_model(self, model_path):
        """Load ONNX model for inference"""
        if not INFERENCE_AVAILABLE:
            logger.error("Inference libraries not available")
            return False
            
        try:
            logger.info(f"Loading ONNX model from {model_path}")
            
            # Load tokenizer
            tokenizer_path = os.path.dirname(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load ONNX model
            onnx_model_path = os.path.join(model_path, "model.onnx") if os.path.isdir(model_path) else model_path
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(onnx_model_path, providers=providers)
            
            logger.info("ONNX model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return False
    
    def generate_pytorch(self, 
                        prompt, 
                        temperature=0.7, 
                        max_length=100, 
                        top_p=0.9, 
                        top_k=50,
                        stream=False):
        """Generate text using PyTorch model"""
        if self.model is None or self.tokenizer is None:
            logger.error("PyTorch model not loaded")
            return None
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Set up streamer if requested
            streamer = TextStreamer(self.tokenizer, skip_prompt=True) if stream else None
            
            # Generate
            with torch.no_grad():
                start_time = time.time()
                
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs),
                    streamer=streamer
                )
                
                generation_time = time.time() - start_time
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            return {
                'text': response,
                'generation_time': generation_time,
                'tokens_generated': len(outputs[0]) - len(inputs[0])
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None
    
    def generate_onnx(self, 
                     prompt, 
                     temperature=0.7, 
                     max_length=100, 
                     top_p=0.9, 
                     top_k=50):
        """Generate text using ONNX model"""
        if self.onnx_session is None or self.tokenizer is None:
            logger.error("ONNX model not loaded")
            return None
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="np")
            
            start_time = time.time()
            
            # Simple greedy generation for ONNX (more complex sampling would require custom implementation)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            generated_tokens = []
            
            for _ in range(max_length):
                # Run inference
                outputs = self.onnx_session.run(None, {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                })
                
                # Get logits
                logits = outputs[0]
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Get next token (greedy for simplicity)
                next_token = np.argmax(logits[0, -1, :])
                
                # Check for end token
                if next_token == self.tokenizer.eos_token_id:
                    break
                
                # Add token to sequence
                generated_tokens.append(next_token)
                
                # Update inputs for next iteration
                new_token = np.array([[next_token]])
                input_ids = np.concatenate([input_ids, new_token], axis=1)
                attention_mask = np.concatenate([attention_mask, np.ones((1, 1))], axis=1)
            
            generation_time = time.time() - start_time
            
            # Decode response
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return {
                'text': response,
                'generation_time': generation_time,
                'tokens_generated': len(generated_tokens)
            }
            
        except Exception as e:
            logger.error(f"ONNX generation failed: {e}")
            return None
    
    def generate(self, 
                prompt, 
                model_type="pytorch",
                temperature=None,
                max_length=None,
                top_p=None,
                top_k=None,
                stream=False):
        """Generate text using specified model type"""
        
        # Use defaults from config
        temperature = temperature or self.config['api']['generation']['default_temperature']
        max_length = max_length or self.config['api']['generation']['default_max_length']
        top_p = top_p or self.config['api']['generation']['default_top_p']
        top_k = top_k or self.config['api']['generation']['default_top_k']
        
        logger.info(f"Generating text with {model_type} model")
        logger.info(f"Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        logger.info(f"Parameters: temp={temperature}, max_len={max_length}, top_p={top_p}, top_k={top_k}")
        
        if model_type == "pytorch":
            return self.generate_pytorch(prompt, temperature, max_length, top_p, top_k, stream)
        elif model_type == "onnx":
            return self.generate_onnx(prompt, temperature, max_length, top_p, top_k)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
    
    def interactive_generation(self, model_type="pytorch"):
        """Interactive text generation session"""
        print(f"\nInteractive Text Generation ({model_type} model)")
        print("Type 'quit' to exit, 'clear' to clear conversation")
        print("=" * 50)
        
        conversation = ""
        
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                elif prompt.lower() in ['clear', 'c']:
                    conversation = ""
                    print("Conversation cleared.")
                    continue
                elif not prompt:
                    continue
                
                # Add to conversation
                full_prompt = conversation + prompt if conversation else prompt
                
                # Generate response
                result = self.generate(full_prompt, model_type=model_type, stream=True)
                
                if result:
                    response = result['text']
                    print(f"\nResponse: {response}")
                    print(f"Generation time: {result['generation_time']:.2f}s")
                    print(f"Tokens generated: {result['tokens_generated']}")
                    
                    # Update conversation
                    conversation = full_prompt + " " + response + " "
                else:
                    print("Generation failed!")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Generate text using trained models")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--type", default="pytorch", choices=["pytorch", "onnx"], 
                       help="Model type")
    parser.add_argument("--prompt", help="Text prompt for generation")
    parser.add_argument("--temperature", type=float, help="Generation temperature")
    parser.add_argument("--max-length", type=int, help="Maximum generation length")
    parser.add_argument("--top-p", type=float, help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, help="Top-k sampling parameter")
    parser.add_argument("--stream", action="store_true", help="Stream output tokens")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--input-file", help="Read prompts from file")
    parser.add_argument("--output-file", help="Save results to file")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = TextGenerator()
    
    # Load model
    if args.type == "pytorch":
        if not generator.load_pytorch_model(args.model):
            sys.exit(1)
    elif args.type == "onnx":
        if not generator.load_onnx_model(args.model):
            sys.exit(1)
    
    # Interactive mode
    if args.interactive:
        generator.interactive_generation(args.type)
        return
    
    # Batch processing from file
    if args.input_file:
        results = []
        with open(args.input_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        for prompt in prompts:
            result = generator.generate(
                prompt,
                model_type=args.type,
                temperature=args.temperature,
                max_length=args.max_length,
                top_p=args.top_p,
                top_k=args.top_k,
                stream=args.stream
            )
            
            if result:
                results.append({
                    'prompt': prompt,
                    'response': result['text'],
                    'generation_time': result['generation_time'],
                    'tokens_generated': result['tokens_generated']
                })
                print(f"Prompt: {prompt}")
                print(f"Response: {result['text']}")
                print("-" * 50)
        
        # Save results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        return
    
    # Single prompt generation
    if args.prompt:
        result = generator.generate(
            args.prompt,
            model_type=args.type,
            temperature=args.temperature,
            max_length=args.max_length,
            top_p=args.top_p,
            top_k=args.top_k,
            stream=args.stream
        )
        
        if result:
            print(f"Prompt: {args.prompt}")
            print(f"Response: {result['text']}")
            print(f"Generation time: {result['generation_time']:.2f}s")
            print(f"Tokens generated: {result['tokens_generated']}")
            
            # Save to file if specified
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    json.dump(result, f, indent=2)
        else:
            print("Generation failed!")
            sys.exit(1)
    else:
        print("No prompt specified. Use --prompt, --input-file, or --interactive")
        sys.exit(1)

if __name__ == "__main__":
    main()