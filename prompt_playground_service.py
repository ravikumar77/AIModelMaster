"""
Prompt Playground Service - Advanced prompt crafting, testing, and management
"""
import json
import yaml
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from app import db
from models import (
    PromptTemplate, PromptSession, PromptGeneration, PromptExport,
    LLMModel, User
)
from llm_service import LLMService

logger = logging.getLogger(__name__)

class PromptPlaygroundService:
    def __init__(self):
        self.llm_service = LLMService()
        
    # Template Management
    def create_template(self, name: str, template_content: str, description: str = "", 
                       category: str = "general", tags: List[str] = None, 
                       is_public: bool = False, created_by: int = None) -> PromptTemplate:
        """Create a new prompt template"""
        try:
            template = PromptTemplate(
                name=name,
                description=description,
                template_content=template_content,
                category=category,
                tags=json.dumps(tags or []),
                is_public=is_public,
                created_by=created_by
            )
            db.session.add(template)
            db.session.commit()
            
            logger.info(f"Created prompt template: {name}")
            return template
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating template: {e}")
            raise
    
    def get_templates(self, category: str = None, user_id: int = None, 
                     include_public: bool = True) -> List[PromptTemplate]:
        """Get prompt templates with filtering"""
        query = PromptTemplate.query
        
        if category:
            query = query.filter_by(category=category)
            
        if user_id:
            if include_public:
                query = query.filter(
                    (PromptTemplate.created_by == user_id) | 
                    (PromptTemplate.is_public == True)
                )
            else:
                query = query.filter_by(created_by=user_id)
        elif include_public:
            query = query.filter_by(is_public=True)
            
        return query.order_by(PromptTemplate.usage_count.desc()).all()
    
    def update_template_usage(self, template_id: int):
        """Increment template usage count"""
        template = PromptTemplate.query.get(template_id)
        if template:
            template.usage_count += 1
            db.session.commit()
    
    # Session Management
    def create_session(self, name: str, prompt_text: str, model_id: int,
                      template_id: int = None, created_by: int = None,
                      parameters: Dict[str, Any] = None) -> PromptSession:
        """Create a new prompt session"""
        try:
            params = parameters or {}
            session = PromptSession(
                name=name,
                prompt_text=prompt_text,
                model_id=model_id,
                template_id=template_id,
                created_by=created_by,
                temperature=params.get('temperature', 0.7),
                max_length=params.get('max_length', 100),
                top_p=params.get('top_p', 0.9),
                top_k=params.get('top_k', 50),
                repetition_penalty=params.get('repetition_penalty', 1.0),
                context_messages=json.dumps(params.get('context_messages', [])),
                few_shot_examples=json.dumps(params.get('few_shot_examples', [])),
                tags=json.dumps(params.get('tags', []))
            )
            db.session.add(session)
            db.session.commit()
            
            logger.info(f"Created prompt session: {name}")
            return session
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating session: {e}")
            raise
    
    def get_sessions(self, user_id: int = None, include_favorites: bool = False) -> List[PromptSession]:
        """Get prompt sessions"""
        query = PromptSession.query
        
        if user_id:
            query = query.filter_by(created_by=user_id)
            
        if include_favorites:
            query = query.filter_by(is_favorite=True)
            
        return query.order_by(PromptSession.updated_at.desc()).all()
    
    def update_session(self, session_id: int, **kwargs) -> PromptSession:
        """Update session parameters"""
        session = PromptSession.query.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
            
        for key, value in kwargs.items():
            if hasattr(session, key):
                setattr(session, key, value)
                
        session.updated_at = datetime.utcnow()
        db.session.commit()
        return session
    
    def toggle_favorite(self, session_id: int) -> bool:
        """Toggle session favorite status"""
        session = PromptSession.query.get(session_id)
        if session:
            session.is_favorite = not session.is_favorite
            db.session.commit()
            return session.is_favorite
        return False
    
    # Text Generation
    def generate_text(self, session_id: int, input_text: str = None, 
                     override_params: Dict[str, Any] = None) -> PromptGeneration:
        """Generate text using session configuration"""
        session = PromptSession.query.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
            
        # Build full prompt
        full_prompt = self._build_full_prompt(session, input_text)
        
        # Get generation parameters
        params = self._get_generation_params(session, override_params)
        
        # Generate text
        start_time = time.time()
        try:
            result = self.llm_service.generate_text(
                model_id=session.model_id,
                prompt=full_prompt,
                **params
            )
            generation_time = time.time() - start_time
            
            # Calculate tokens per second (mock calculation)
            tokens_generated = len(result.split())
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            # Save generation
            generation = PromptGeneration(
                session_id=session_id,
                input_text=input_text or "",
                generated_text=result,
                full_prompt=full_prompt,
                generation_time=generation_time,
                tokens_generated=tokens_generated,
                tokens_per_second=tokens_per_second,
                temperature=params.get('temperature'),
                max_length=params.get('max_length'),
                top_p=params.get('top_p'),
                top_k=params.get('top_k')
            )
            db.session.add(generation)
            db.session.commit()
            
            logger.info(f"Generated text for session {session_id}: {tokens_generated} tokens in {generation_time:.2f}s")
            return generation
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            # Create error generation record
            generation = PromptGeneration(
                session_id=session_id,
                input_text=input_text or "",
                generated_text=f"Error: {str(e)}",
                full_prompt=full_prompt,
                generation_time=time.time() - start_time
            )
            db.session.add(generation)
            db.session.commit()
            return generation
    
    def _build_full_prompt(self, session: PromptSession, input_text: str = None) -> str:
        """Build complete prompt with context and examples"""
        prompt_parts = []
        
        # Add few-shot examples
        if session.few_shot_examples:
            examples = json.loads(session.few_shot_examples)
            for example in examples:
                if isinstance(example, dict):
                    prompt_parts.append(f"Input: {example.get('input', '')}")
                    prompt_parts.append(f"Output: {example.get('output', '')}")
                else:
                    prompt_parts.append(str(example))
            prompt_parts.append("")  # Empty line separator
        
        # Add context messages
        if session.context_messages:
            messages = json.loads(session.context_messages)
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    prompt_parts.append(f"{role.title()}: {content}")
                else:
                    prompt_parts.append(str(msg))
            prompt_parts.append("")  # Empty line separator
        
        # Add main prompt
        prompt_parts.append(session.prompt_text)
        
        # Add current input
        if input_text:
            prompt_parts.append(input_text)
        
        return "\n".join(prompt_parts)
    
    def _get_generation_params(self, session: PromptSession, 
                              override_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get generation parameters with optional overrides"""
        params = {
            'temperature': session.temperature,
            'max_length': session.max_length,
            'top_p': session.top_p,
            'top_k': session.top_k
        }
        
        if override_params:
            params.update(override_params)
            
        return params
    
    # Rating and Feedback
    def rate_generation(self, generation_id: int, rating: int, flag_reason: str = None) -> bool:
        """Rate a generation (1-5 stars) and optionally flag it"""
        generation = PromptGeneration.query.get(generation_id)
        if not generation:
            return False
            
        generation.user_rating = max(1, min(5, rating))
        
        if flag_reason:
            generation.is_flagged = True
            generation.flag_reason = flag_reason
            
        db.session.commit()
        return True
    
    # Export Functionality
    def export_session(self, session_id: int, export_format: str) -> PromptExport:
        """Export session configuration in various formats"""
        session = PromptSession.query.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
            
        export_content = self._generate_export_content(session, export_format)
        
        export = PromptExport(
            session_id=session_id,
            export_format=export_format,
            export_content=export_content
        )
        db.session.add(export)
        db.session.commit()
        
        return export
    
    def _generate_export_content(self, session: PromptSession, format_type: str) -> str:
        """Generate export content in specified format"""
        session_data = {
            'name': session.name,
            'prompt_text': session.prompt_text,
            'model_id': session.model_id,
            'parameters': {
                'temperature': session.temperature,
                'max_length': session.max_length,
                'top_p': session.top_p,
                'top_k': session.top_k,
                'repetition_penalty': session.repetition_penalty
            },
            'context_messages': json.loads(session.context_messages or '[]'),
            'few_shot_examples': json.loads(session.few_shot_examples or '[]'),
            'tags': json.loads(session.tags or '[]')
        }
        
        if format_type == 'json':
            return json.dumps(session_data, indent=2)
        elif format_type == 'yaml':
            return yaml.dump(session_data, default_flow_style=False)
        elif format_type == 'curl':
            return self._generate_curl_command(session_data)
        elif format_type == 'python':
            return self._generate_python_code(session_data)
        elif format_type == 'triton':
            return self._generate_triton_config(session_data)
        elif format_type == 'tensorflow_lite':
            return self._generate_tflite_config(session_data)
        elif format_type == 'huggingface':
            return self._generate_huggingface_config(session_data)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _generate_curl_command(self, session_data: Dict) -> str:
        """Generate cURL command for API call"""
        curl_template = """curl -X POST "http://localhost:5000/api/models/{model_id}/generate" \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: YOUR_API_KEY" \\
  -d '{data}'"""
        
        request_data = {
            "prompt": session_data['prompt_text'],
            "temperature": session_data['parameters']['temperature'],
            "max_length": session_data['parameters']['max_length'],
            "top_p": session_data['parameters']['top_p'],
            "top_k": session_data['parameters']['top_k']
        }
        
        return curl_template.format(
            model_id=session_data['model_id'],
            data=json.dumps(request_data)
        )
    
    def _generate_python_code(self, session_data: Dict) -> str:
        """Generate Python code for API call"""
        python_template = """import requests
import json

# LLM Platform API Configuration
API_BASE_URL = "http://localhost:5000/api"
API_KEY = "YOUR_API_KEY"

# Request parameters
data = {data}

# Make API call
headers = {{
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}}

response = requests.post(
    f"{{API_BASE_URL}}/models/{model_id}/generate",
    headers=headers,
    json=data
)

if response.status_code == 200:
    result = response.json()
    print("Generated text:", result['text'])
else:
    print("Error:", response.text)"""
        
        request_data = {
            "prompt": session_data['prompt_text'],
            "temperature": session_data['parameters']['temperature'],
            "max_length": session_data['parameters']['max_length'],
            "top_p": session_data['parameters']['top_p'],
            "top_k": session_data['parameters']['top_k']
        }
        
        return python_template.format(
            model_id=session_data['model_id'],
            data=json.dumps(request_data, indent=4)
        )
    
    def _generate_triton_config(self, session_data: Dict) -> str:
        """Generate Triton Inference Server configuration"""
        triton_config = f"""# Triton Model Configuration for {session_data['name']}
name: "llm_model_{session_data['model_id']}"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {{
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }}
]
output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  }}
]

# Model Parameters
parameters: {{
  "temperature": {{
    string_value: "{session_data['parameters']['temperature']}"
  }},
  "max_length": {{
    string_value: "{session_data['parameters']['max_length']}"
  }},
  "top_p": {{
    string_value: "{session_data['parameters']['top_p']}"
  }},
  "top_k": {{
    string_value: "{session_data['parameters']['top_k']}"
  }}
}}

# Dynamic Batching
dynamic_batching {{
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 5000
}}

# Model Repository Structure
# models/
#   llm_model_{session_data['model_id']}/
#     config.pbtxt  (this file)
#     1/
#       model.pt    (your PyTorch model)

# Deployment Script
"""

        deployment_script = f"""#!/bin/bash
# Triton Server Deployment Script

# 1. Prepare model repository
mkdir -p models/llm_model_{session_data['model_id']}/1

# 2. Copy your model file
cp your_model.pt models/llm_model_{session_data['model_id']}/1/model.pt

# 3. Start Triton Server
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \\
  -v ${{PWD}}/models:/models \\
  nvcr.io/nvidia/tritonserver:22.12-py3 \\
  tritonserver --model-repository=/models

# 4. Test inference
curl -X POST localhost:8000/v2/models/llm_model_{session_data['model_id']}/infer \\
  -H "Content-Type: application/json" \\
  -d '{{
    "inputs": [
      {{
        "name": "input_ids",
        "shape": [1, -1],
        "datatype": "INT32",
        "data": [/* your tokenized input */]
      }}
    ]
  }}'
"""
        
        return triton_config + "\n\n# " + "="*50 + "\n# DEPLOYMENT SCRIPT\n# " + "="*50 + "\n\n" + deployment_script
    
    def _generate_tflite_config(self, session_data: Dict) -> str:
        """Generate TensorFlow Lite conversion and deployment configuration"""
        tflite_config = f"""# TensorFlow Lite Export Configuration for {session_data['name']}

# Python Conversion Script
import tensorflow as tf
import numpy as np

def convert_to_tflite():
    \"\"\"Convert PyTorch model to TensorFlow Lite\"\"\"
    
    # Step 1: Load your PyTorch model
    # model = torch.load('your_model.pth')
    # model.eval()
    
    # Step 2: Convert to ONNX first
    # torch.onnx.export(model, dummy_input, 'model.onnx')
    
    # Step 3: Convert ONNX to TensorFlow
    # import onnx_tf
    # onnx_model = onnx.load('model.onnx')
    # tf_rep = onnx_tf.backend.prepare(onnx_model)
    # tf_rep.export_graph('model.pb')
    
    # Step 4: Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_dir')
    
    # Optimization settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Generation parameters as metadata
    converter.experimental_new_converter = True
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    return tflite_model

# Model Parameters (embed in metadata)
GENERATION_PARAMS = {{
    'temperature': {session_data['parameters']['temperature']},
    'max_length': {session_data['parameters']['max_length']},
    'top_p': {session_data['parameters']['top_p']},
    'top_k': {session_data['parameters']['top_k']},
    'prompt_template': '''{session_data['prompt_text']}'''
}}

# Mobile Deployment (Android/iOS)
"""

        mobile_code = f"""
// Android Kotlin Implementation
class LLMInference(context: Context) {{
    private val interpreter: Interpreter
    
    init {{
        val model = loadModelFile(context, "model.tflite")
        interpreter = Interpreter(model)
    }}
    
    fun generateText(inputText: String): String {{
        // Tokenize input
        val inputIds = tokenize(inputText)
        
        // Prepare input tensor
        val inputArray = Array(1) {{ inputIds.toIntArray() }}
        
        // Prepare output tensor
        val outputArray = Array(1) {{ FloatArray(vocab_size) }}
        
        // Run inference
        interpreter.run(inputArray, outputArray)
        
        // Decode output
        return decode(outputArray[0])
    }}
    
    companion object {{
        const val TEMPERATURE = {session_data['parameters']['temperature']}f
        const val MAX_LENGTH = {session_data['parameters']['max_length']}
        const val TOP_P = {session_data['parameters']['top_p']}f
    }}
}}

// iOS Swift Implementation
class LLMInference {{
    private var interpreter: Interpreter?
    
    init() {{
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "tflite") else {{
            fatalError("Failed to load model")
        }}
        
        do {{
            interpreter = try Interpreter(modelPath: modelPath)
            try interpreter?.allocateTensors()
        }} catch {{
            print("Failed to create interpreter: \\(error)")
        }}
    }}
    
    func generateText(inputText: String) -> String {{
        // Implementation similar to Android
        let inputIds = tokenize(inputText)
        
        do {{
            try interpreter?.copy(Data(bytes: inputIds), toInputAt: 0)
            try interpreter?.invoke()
            let outputTensor = try interpreter?.output(at: 0)
            
            return decode(outputTensor?.data)
        }} catch {{
            print("Inference failed: \\(error)")
            return ""
        }}
    }}
}}
"""
        
        return tflite_config + "\n\n# " + "="*50 + "\n# MOBILE DEPLOYMENT CODE\n# " + "="*50 + "\n\n" + mobile_code
    
    def _generate_huggingface_config(self, session_data: Dict) -> str:
        """Generate HuggingFace Hub deployment configuration"""
        hf_config = f"""# HuggingFace Hub Deployment for {session_data['name']}

# Model Configuration (config.json)
{{
  "model_type": "gpt2",
  "architectures": ["GPT2LMHeadModel"],
  "vocab_size": 50257,
  "n_positions": 1024,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_layer": 12,
  "n_head": 12,
  "activation_function": "gelu_new",
  "resid_pdrop": 0.1,
  "embd_pdrop": 0.1,
  "attn_pdrop": 0.1,
  "layer_norm_epsilon": 1e-05,
  "initializer_range": 0.02,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "summary_activation": null,
  "summary_proj_to_labels": true,
  "summary_first_dropout": 0.1,
  "bos_token_id": 50256,
  "eos_token_id": 50256,
  "transformers_version": "4.21.0",
  "task_specific_params": {{
    "text-generation": {{
      "do_sample": true,
      "temperature": {session_data['parameters']['temperature']},
      "max_length": {session_data['parameters']['max_length']},
      "top_p": {session_data['parameters']['top_p']},
      "top_k": {session_data['parameters']['top_k']}
    }}
  }}
}}

# Model Card (README.md)
---
language: en
license: mit
tags:
- text-generation
- gpt2
- fine-tuned
widget:
- text: "{session_data['prompt_text'][:100]}..."
---

# {session_data['name']} - Fine-tuned Language Model

## Model Description
This model is a fine-tuned version of GPT-2 optimized for specific text generation tasks.

## Intended Use
- **Primary Use**: {', '.join(json.loads(session_data['tags']) if session_data['tags'] else ['general text generation'])}
- **Out-of-scope**: This model should not be used for harmful content generation

## Training Data
- Base Model: GPT-2
- Fine-tuning: Custom dataset

## Training Procedure
- Temperature: {session_data['parameters']['temperature']}
- Max Length: {session_data['parameters']['max_length']}
- Top-p: {session_data['parameters']['top_p']}
- Top-k: {session_data['parameters']['top_k']}

## Usage

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("your-username/{session_data['name'].lower().replace(' ', '-')}")
tokenizer = GPT2Tokenizer.from_pretrained("your-username/{session_data['name'].lower().replace(' ', '-')}")

# Generate text
input_text = "{session_data['prompt_text'][:50]}..."
inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(
    inputs,
    max_length={session_data['parameters']['max_length']},
    temperature={session_data['parameters']['temperature']},
    top_p={session_data['parameters']['top_p']},
    top_k={session_data['parameters']['top_k']},
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Deployment Script
"""

        deployment_script = f"""#!/bin/bash
# HuggingFace Hub Upload Script

# Install required packages
pip install huggingface_hub transformers torch

# Login to HuggingFace (requires token)
huggingface-cli login

# Python upload script
python << EOF
from huggingface_hub import Repository, HfApi
import torch
import json

# Model information
model_name = "{session_data['name'].lower().replace(' ', '-')}"
username = "your-username"  # Replace with your HF username
repo_name = f"{{username}}/{{model_name}}"

# Create repository
api = HfApi()
try:
    api.create_repo(repo_id=repo_name, private=False)
    print(f"Created repository: {{repo_name}}")
except Exception as e:
    print(f"Repository might already exist: {{e}}")

# Clone repository
repo = Repository(
    local_dir=f"./{{model_name}}",
    clone_from=repo_name,
    use_auth_token=True
)

# Save model files
# torch.save(your_model.state_dict(), f"./{{model_name}}/pytorch_model.bin")

# Save config
config = {{
    "model_type": "gpt2",
    "vocab_size": 50257,
    "task_specific_params": {{
        "text-generation": {{
            "temperature": {session_data['parameters']['temperature']},
            "max_length": {session_data['parameters']['max_length']},
            "top_p": {session_data['parameters']['top_p']},
            "top_k": {session_data['parameters']['top_k']}
        }}
    }}
}}

with open(f"./{{model_name}}/config.json", "w") as f:
    json.dump(config, f, indent=2)

# Commit and push
repo.git_add()
repo.git_commit("Add fine-tuned model")
repo.git_push()

print(f"Model uploaded successfully to: https://huggingface.co/{{repo_name}}")
EOF

# Test deployment
python << EOF
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{username}/{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{username}/{model_name}")

# Test generation
text = "{session_data['prompt_text'][:50]}..."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
EOF
"""
        
        return hf_config + "\n\n# " + "="*50 + "\n# DEPLOYMENT SCRIPT\n# " + "="*50 + "\n\n" + deployment_script
    
    # Analytics and Statistics
    def get_session_analytics(self, session_id: int) -> Dict[str, Any]:
        """Get analytics for a prompt session"""
        session = PromptSession.query.get(session_id)
        if not session:
            return {}
            
        generations = PromptGeneration.query.filter_by(session_id=session_id).all()
        
        if not generations:
            return {'total_generations': 0}
            
        total_generations = len(generations)
        total_tokens = sum(g.tokens_generated or 0 for g in generations)
        avg_generation_time = sum(g.generation_time or 0 for g in generations) / total_generations
        avg_tokens_per_second = sum(g.tokens_per_second or 0 for g in generations) / total_generations
        
        ratings = [g.user_rating for g in generations if g.user_rating]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        return {
            'total_generations': total_generations,
            'total_tokens': total_tokens,
            'avg_generation_time': round(avg_generation_time, 2),
            'avg_tokens_per_second': round(avg_tokens_per_second, 2),
            'avg_rating': round(avg_rating, 2),
            'flagged_count': sum(1 for g in generations if g.is_flagged),
            'last_generated': max(g.created_at for g in generations).isoformat()
        }
    
    # Default Templates Creation
    def initialize_default_templates(self):
        """Create default prompt templates"""
        default_templates = [
            {
                'name': 'Code Generation',
                'description': 'Template for generating code snippets',
                'template_content': 'Write a Python function that {task}:\n\n```python\n',
                'category': 'coding',
                'tags': ['python', 'code', 'function'],
                'is_public': True
            },
            {
                'name': 'Creative Writing',
                'description': 'Template for creative writing tasks',
                'template_content': 'Write a creative story about {topic}. The story should be {tone} and approximately {length} words.\n\nStory:',
                'category': 'creative',
                'tags': ['story', 'creative', 'writing'],
                'is_public': True
            },
            {
                'name': 'Question Answering',
                'description': 'Template for answering questions',
                'template_content': 'Answer the following question accurately and concisely:\n\nQuestion: {question}\n\nAnswer:',
                'category': 'general',
                'tags': ['qa', 'question', 'answer'],
                'is_public': True
            },
            {
                'name': 'Text Summarization',
                'description': 'Template for summarizing text',
                'template_content': 'Summarize the following text in {length} sentences:\n\n{text}\n\nSummary:',
                'category': 'general',
                'tags': ['summary', 'text', 'condensation'],
                'is_public': True
            },
            {
                'name': 'Code Review',
                'description': 'Template for code review and suggestions',
                'template_content': 'Review the following code and provide suggestions for improvement:\n\n```{language}\n{code}\n```\n\nReview:',
                'category': 'coding',
                'tags': ['review', 'code', 'improvement'],
                'is_public': True
            }
        ]
        
        for template_data in default_templates:
            existing = PromptTemplate.query.filter_by(name=template_data['name']).first()
            if not existing:
                self.create_template(**template_data)
                
        logger.info("Default prompt templates initialized")

# Global service instance
prompt_playground_service = PromptPlaygroundService()