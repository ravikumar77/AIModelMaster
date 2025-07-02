"""
Custom Dataset Service - Handle user dataset uploads and processing
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from werkzeug.utils import secure_filename
from app import db
from models import CustomDataset, DatasetFormat, User

class CustomDatasetService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.upload_folder = 'data/user_datasets'
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.allowed_extensions = {'.txt', '.jsonl', '.json', '.csv'}
        
        # Ensure upload directory exists
        os.makedirs(self.upload_folder, exist_ok=True)
    
    def upload_dataset(self, file_data, filename: str, dataset_name: str, 
                      description: str, dataset_format: str, user_id: int) -> Dict:
        """Upload and process a custom dataset"""
        try:
            # Validate file
            if not self._is_valid_file(filename):
                return {"success": False, "error": "Invalid file format"}
            
            # Secure filename
            secure_name = secure_filename(filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_filename = f"{user_id}_{timestamp}_{secure_name}"
            file_path = os.path.join(self.upload_folder, final_filename)
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            file_size = len(file_data)
            
            # Process dataset based on format
            format_enum = DatasetFormat[dataset_format.upper()]
            processing_result = self._process_dataset(file_path, format_enum)
            
            # Create database record
            dataset = CustomDataset(
                name=dataset_name,
                description=description,
                dataset_format=format_enum,
                original_filename=filename,
                file_path=file_path,
                file_size=file_size,
                created_by=user_id,
                num_samples=processing_result.get('num_samples', 0),
                sample_length_avg=processing_result.get('avg_length', 0.0),
                sample_length_max=processing_result.get('max_length', 0),
                is_processed=processing_result.get('success', False),
                processing_logs=processing_result.get('logs', ''),
                validation_errors=processing_result.get('errors', '')
            )
            
            db.session.add(dataset)
            db.session.commit()
            
            return {
                "success": True,
                "dataset_id": dataset.id,
                "message": "Dataset uploaded successfully",
                "processing_result": processing_result
            }
            
        except Exception as e:
            self.logger.error(f"Error uploading dataset: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _is_valid_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return any(filename.lower().endswith(ext) for ext in self.allowed_extensions)
    
    def _process_dataset(self, file_path: str, dataset_format: DatasetFormat) -> Dict:
        """Process and validate dataset based on format"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if dataset_format == DatasetFormat.TEXT:
                return self._process_text_dataset(content)
            elif dataset_format == DatasetFormat.JSONL:
                return self._process_jsonl_dataset(content)
            elif dataset_format == DatasetFormat.CSV:
                return self._process_csv_dataset(content)
            elif dataset_format == DatasetFormat.CONVERSATION:
                return self._process_conversation_dataset(content)
            elif dataset_format == DatasetFormat.INSTRUCTION:
                return self._process_instruction_dataset(content)
            else:
                return {"success": False, "errors": "Unsupported format"}
                
        except Exception as e:
            return {"success": False, "errors": str(e)}
    
    def _process_text_dataset(self, content: str) -> Dict:
        """Process plain text dataset"""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        lengths = [len(line) for line in lines]
        
        return {
            "success": True,
            "num_samples": len(lines),
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "logs": f"Processed {len(lines)} text samples"
        }
    
    def _process_jsonl_dataset(self, content: str) -> Dict:
        """Process JSONL dataset"""
        lines = content.strip().split('\n')
        valid_samples = []
        errors = []
        
        for i, line in enumerate(lines):
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    valid_samples.append(data)
                else:
                    errors.append(f"Line {i+1}: Not a JSON object")
            except json.JSONDecodeError as e:
                errors.append(f"Line {i+1}: {str(e)}")
        
        # Calculate text lengths
        lengths = []
        for sample in valid_samples:
            if 'text' in sample:
                lengths.append(len(sample['text']))
            elif 'input' in sample and 'output' in sample:
                lengths.append(len(sample['input']) + len(sample['output']))
        
        return {
            "success": len(valid_samples) > 0,
            "num_samples": len(valid_samples),
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "logs": f"Processed {len(valid_samples)} valid JSONL samples",
            "errors": "; ".join(errors) if errors else ""
        }
    
    def _process_csv_dataset(self, content: str) -> Dict:
        """Process CSV dataset"""
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return {"success": False, "errors": "CSV must have header and at least one data row"}
        
        # Simple CSV processing (assumes text column exists)
        header = lines[0].split(',')
        data_rows = lines[1:]
        
        # Look for text-like columns
        text_columns = [col.strip('"') for col in header if 'text' in col.lower() or 'content' in col.lower()]
        
        lengths = []
        for row in data_rows:
            columns = row.split(',')
            if len(columns) >= len(header):
                for col_text in columns:
                    if col_text.strip():
                        lengths.append(len(col_text.strip('"')))
        
        return {
            "success": True,
            "num_samples": len(data_rows),
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "logs": f"Processed {len(data_rows)} CSV rows with {len(header)} columns"
        }
    
    def _process_conversation_dataset(self, content: str) -> Dict:
        """Process conversation format dataset"""
        try:
            data = json.loads(content)
            if not isinstance(data, list):
                return {"success": False, "errors": "Conversation dataset must be a JSON array"}
            
            valid_conversations = []
            errors = []
            
            for i, conv in enumerate(data):
                if not isinstance(conv, dict) or 'messages' not in conv:
                    errors.append(f"Conversation {i+1}: Missing 'messages' field")
                    continue
                
                if not isinstance(conv['messages'], list):
                    errors.append(f"Conversation {i+1}: 'messages' must be an array")
                    continue
                
                valid_conversations.append(conv)
            
            # Calculate lengths
            lengths = []
            for conv in valid_conversations:
                total_length = sum(len(msg.get('content', '')) for msg in conv['messages'])
                lengths.append(total_length)
            
            return {
                "success": len(valid_conversations) > 0,
                "num_samples": len(valid_conversations),
                "avg_length": sum(lengths) / len(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "logs": f"Processed {len(valid_conversations)} conversation samples",
                "errors": "; ".join(errors) if errors else ""
            }
            
        except json.JSONDecodeError as e:
            return {"success": False, "errors": f"Invalid JSON: {str(e)}"}
    
    def _process_instruction_dataset(self, content: str) -> Dict:
        """Process instruction-following dataset"""
        try:
            data = json.loads(content)
            if not isinstance(data, list):
                return {"success": False, "errors": "Instruction dataset must be a JSON array"}
            
            valid_instructions = []
            errors = []
            
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    errors.append(f"Item {i+1}: Must be a JSON object")
                    continue
                
                if 'instruction' not in item:
                    errors.append(f"Item {i+1}: Missing 'instruction' field")
                    continue
                
                if 'response' not in item and 'output' not in item:
                    errors.append(f"Item {i+1}: Missing 'response' or 'output' field")
                    continue
                
                valid_instructions.append(item)
            
            # Calculate lengths
            lengths = []
            for item in valid_instructions:
                instruction_len = len(item.get('instruction', ''))
                response_len = len(item.get('response', item.get('output', '')))
                lengths.append(instruction_len + response_len)
            
            return {
                "success": len(valid_instructions) > 0,
                "num_samples": len(valid_instructions),
                "avg_length": sum(lengths) / len(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "logs": f"Processed {len(valid_instructions)} instruction samples",
                "errors": "; ".join(errors) if errors else ""
            }
            
        except json.JSONDecodeError as e:
            return {"success": False, "errors": f"Invalid JSON: {str(e)}"}
    
    def get_user_datasets(self, user_id: int) -> List[CustomDataset]:
        """Get all datasets for a user"""
        return CustomDataset.query.filter_by(created_by=user_id).order_by(CustomDataset.created_at.desc()).all()
    
    def get_dataset(self, dataset_id: int, user_id: int = None) -> Optional[CustomDataset]:
        """Get a specific dataset"""
        query = CustomDataset.query.filter_by(id=dataset_id)
        if user_id:
            query = query.filter_by(created_by=user_id)
        return query.first()
    
    def delete_dataset(self, dataset_id: int, user_id: int) -> Dict:
        """Delete a dataset"""
        try:
            dataset = self.get_dataset(dataset_id, user_id)
            if not dataset:
                return {"success": False, "error": "Dataset not found"}
            
            # Remove file
            if os.path.exists(dataset.file_path):
                os.remove(dataset.file_path)
            
            # Remove from database
            db.session.delete(dataset)
            db.session.commit()
            
            return {"success": True, "message": "Dataset deleted successfully"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_dataset_sample(self, dataset_id: int, user_id: int, num_samples: int = 3) -> Dict:
        """Get sample data from dataset for preview"""
        try:
            dataset = self.get_dataset(dataset_id, user_id)
            if not dataset:
                return {"success": False, "error": "Dataset not found"}
            
            with open(dataset.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            samples = []
            if dataset.dataset_format == DatasetFormat.TEXT:
                lines = content.split('\n')[:num_samples]
                samples = [{"text": line} for line in lines if line.strip()]
            
            elif dataset.dataset_format == DatasetFormat.JSONL:
                lines = content.split('\n')[:num_samples]
                for line in lines:
                    try:
                        samples.append(json.loads(line))
                    except:
                        continue
            
            elif dataset.dataset_format in [DatasetFormat.CONVERSATION, DatasetFormat.INSTRUCTION]:
                data = json.loads(content)
                samples = data[:num_samples] if isinstance(data, list) else []
            
            return {
                "success": True,
                "samples": samples,
                "dataset_info": {
                    "name": dataset.name,
                    "format": dataset.dataset_format.value,
                    "num_samples": dataset.num_samples,
                    "file_size": dataset.file_size
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

# Global service instance
custom_dataset_service = CustomDatasetService()