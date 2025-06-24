#!/usr/bin/env python3
"""
Export PyTorch models to ONNX format for optimized inference
Supports various optimization levels and target platforms
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
    import onnx
    import onnxruntime as ort
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from optimum.onnxruntime import ORTModelForCausalLM
    from optimum.exporters.onnx import main_export
    ONNX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ONNX libraries not available: {e}")
    ONNX_AVAILABLE = False

from app import app, db
from models import LLMModel, ModelStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ONNXExporter:
    def __init__(self, config_path="config/model_config.yaml"):
        """Initialize ONNX exporter"""
        self.config = self.load_config(config_path)
        
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
            'export': {
                'onnx': {
                    'opset_version': 11,
                    'optimization_levels': ['basic', 'optimized', 'quantized']
                }
            }
        }
    
    def create_triton_config(self, model_name, max_batch_size=8, max_sequence_length=512):
        """Create Triton Inference Server configuration"""
        config = f"""
name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}
input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1, {max_sequence_length} ]
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1, {max_sequence_length} ]
  }}
]
output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1, {max_sequence_length}, -1 ]
  }}
]
instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
optimization {{
  enable_pinned_input: true
  enable_pinned_output: true
}}
"""
        return config
    
    def export_to_onnx(self,
                       model_path,
                       output_dir="models/onnx",
                       optimization_level="basic",
                       opset_version=None,
                       model_id=None):
        """Export model to ONNX format"""
        
        if not ONNX_AVAILABLE:
            logger.error("ONNX libraries not available")
            return False
        
        opset_version = opset_version or self.config['export']['onnx']['opset_version']
        
        logger.info(f"Exporting model from {model_path} to ONNX")
        logger.info(f"Optimization level: {optimization_level}")
        
        try:
            # Update model status
            if model_id:
                with app.app_context():
                    model_record = LLMModel.query.get(model_id)
                    if model_record:
                        model_record.status = ModelStatus.EXPORTING
                        db.session.commit()
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            logger.info("Model loaded successfully")
            
            # Export based on optimization level
            if optimization_level == "basic":
                success = self._export_basic(model, tokenizer, output_dir, opset_version)
            elif optimization_level == "optimized":
                success = self._export_optimized(model, tokenizer, output_dir, opset_version)
            elif optimization_level == "quantized":
                success = self._export_quantized(model, tokenizer, output_dir, opset_version)
            else:
                logger.error(f"Unknown optimization level: {optimization_level}")
                return False
            
            if success:
                # Generate Triton config
                triton_config = self.create_triton_config(
                    model_name=os.path.basename(output_dir)
                )
                
                with open(os.path.join(output_dir, "config.pbtxt"), 'w') as f:
                    f.write(triton_config)
                
                # Save export metadata
                metadata = {
                    'export_date': datetime.now().isoformat(),
                    'optimization_level': optimization_level,
                    'opset_version': opset_version,
                    'source_model': model_path
                }
                
                with open(os.path.join(output_dir, "export_metadata.json"), 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Export completed successfully. Files saved to {output_dir}")
                
                # Update model status
                if model_id:
                    with app.app_context():
                        model_record = LLMModel.query.get(model_id)
                        if model_record:
                            model_record.status = ModelStatus.AVAILABLE
                            db.session.commit()
                
                return True
            else:
                logger.error("Export failed")
                return False
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            
            # Update model status on failure
            if model_id:
                with app.app_context():
                    model_record = LLMModel.query.get(model_id)
                    if model_record:
                        model_record.status = ModelStatus.ERROR
                        db.session.commit()
            
            return False
    
    def _export_basic(self, model, tokenizer, output_dir, opset_version):
        """Basic ONNX export without optimizations"""
        try:
            # Create dummy inputs
            dummy_input = tokenizer("Hello world", return_tensors="pt")
            
            # Export model
            torch.onnx.export(
                model,
                (dummy_input['input_ids'], dummy_input['attention_mask']),
                os.path.join(output_dir, "model.onnx"),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                    'logits': {0: 'batch_size', 1: 'sequence'}
                }
            )
            
            # Save tokenizer
            tokenizer.save_pretrained(output_dir)
            
            return True
            
        except Exception as e:
            logger.error(f"Basic export failed: {e}")
            return False
    
    def _export_optimized(self, model, tokenizer, output_dir, opset_version):
        """Optimized ONNX export with graph optimizations"""
        try:
            # Use optimum for optimized export
            from optimum.onnxruntime import ORTModelForCausalLM
            
            # Convert to ONNX with optimizations
            ort_model = ORTModelForCausalLM.from_pretrained(
                model,
                export=True,
                provider="CPUExecutionProvider"
            )
            
            # Save optimized model
            ort_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            return True
            
        except Exception as e:
            logger.error(f"Optimized export failed: {e}")
            # Fallback to basic export
            return self._export_basic(model, tokenizer, output_dir, opset_version)
    
    def _export_quantized(self, model, tokenizer, output_dir, opset_version):
        """Quantized ONNX export for smaller model size"""
        try:
            # First do basic export
            if not self._export_basic(model, tokenizer, output_dir, opset_version):
                return False
            
            # Load and quantize the ONNX model
            model_path = os.path.join(output_dir, "model.onnx")
            
            # Dynamic quantization
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantized_model_path = os.path.join(output_dir, "model_quantized.onnx")
            
            quantize_dynamic(
                model_path,
                quantized_model_path,
                weight_type=QuantType.QUInt8
            )
            
            # Replace original with quantized
            os.rename(quantized_model_path, model_path)
            
            logger.info("Model quantized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Quantized export failed: {e}")
            # Fallback to optimized export
            return self._export_optimized(model, tokenizer, output_dir, opset_version)
    
    def validate_onnx_model(self, onnx_path, tokenizer_path=None):
        """Validate exported ONNX model"""
        try:
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            logger.info("ONNX model structure is valid")
            
            # Test inference if tokenizer is available
            if tokenizer_path and os.path.exists(tokenizer_path):
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                
                # Create ONNX Runtime session
                session = ort.InferenceSession(onnx_path)
                
                # Test inference
                test_text = "Hello, world!"
                inputs = tokenizer(test_text, return_tensors="np")
                
                outputs = session.run(None, {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask']
                })
                
                logger.info("ONNX model inference test passed")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"ONNX model validation failed: {e}")
            return False

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--output", default="models/onnx", help="Output directory")
    parser.add_argument("--optimization", default="basic", 
                       choices=["basic", "optimized", "quantized"],
                       help="Optimization level")
    parser.add_argument("--opset-version", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--model-id", type=int, help="Model ID in database")
    parser.add_argument("--validate", action="store_true", help="Validate exported model")
    
    args = parser.parse_args()
    
    exporter = ONNXExporter()
    success = exporter.export_to_onnx(
        model_path=args.model,
        output_dir=args.output,
        optimization_level=args.optimization,
        opset_version=args.opset_version,
        model_id=args.model_id
    )
    
    if success and args.validate:
        onnx_path = os.path.join(args.output, "model.onnx")
        success = exporter.validate_onnx_model(onnx_path, args.output)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()