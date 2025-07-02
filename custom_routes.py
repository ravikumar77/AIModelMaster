"""
Custom Training Routes - API endpoints for custom dataset upload and training
"""
import json
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
from auth_service import require_api_key
from custom_dataset_service import custom_dataset_service
from custom_training_service import custom_training_service
from models import DatasetFormat, TrainingStatus

# Create blueprint
custom_bp = Blueprint('custom', __name__, url_prefix='/custom')

@custom_bp.route('/')
def dashboard():
    """Custom training dashboard"""
    return render_template('custom/dashboard.html')

@custom_bp.route('/datasets')
def datasets():
    """Dataset management page"""
    return render_template('custom/datasets.html')

@custom_bp.route('/training')
def training():
    """Training jobs page"""
    return render_template('custom/training.html')

# Dataset API endpoints
@custom_bp.route('/api/datasets/upload', methods=['POST'])
@require_api_key
def api_upload_dataset():
    """Upload a custom dataset"""
    try:
        user = request.current_user
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        # Get form data
        dataset_name = request.form.get('dataset_name', '').strip()
        description = request.form.get('description', '').strip()
        dataset_format = request.form.get('dataset_format', 'TEXT').upper()
        
        if not dataset_name:
            return jsonify({"success": False, "error": "Dataset name is required"}), 400
        
        if dataset_format not in [f.name for f in DatasetFormat]:
            return jsonify({"success": False, "error": "Invalid dataset format"}), 400
        
        # Read file data
        file_data = file.read()
        if len(file_data) > custom_dataset_service.max_file_size:
            return jsonify({"success": False, "error": "File too large"}), 400
        
        # Upload dataset
        result = custom_dataset_service.upload_dataset(
            file_data=file_data,
            filename=file.filename,
            dataset_name=dataset_name,
            description=description,
            dataset_format=dataset_format,
            user_id=user.id
        )
        
        if result['success']:
            return jsonify(result), 201
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logging.error(f"Error uploading dataset: {str(e)}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

@custom_bp.route('/api/datasets', methods=['GET'])
@require_api_key
def api_list_datasets():
    """List user's custom datasets"""
    try:
        user = request.current_user
        datasets = custom_dataset_service.get_user_datasets(user.id)
        
        return jsonify({
            "success": True,
            "datasets": [dataset.to_dict() for dataset in datasets]
        })
        
    except Exception as e:
        logging.error(f"Error listing datasets: {str(e)}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

@custom_bp.route('/api/datasets/<int:dataset_id>', methods=['GET'])
@require_api_key
def api_get_dataset(dataset_id):
    """Get dataset details"""
    try:
        user = request.current_user
        dataset = custom_dataset_service.get_dataset(dataset_id, user.id)
        
        if not dataset:
            return jsonify({"success": False, "error": "Dataset not found"}), 404
        
        return jsonify({
            "success": True,
            "dataset": dataset.to_dict()
        })
        
    except Exception as e:
        logging.error(f"Error getting dataset: {str(e)}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

@custom_bp.route('/api/datasets/<int:dataset_id>/sample', methods=['GET'])
@require_api_key
def api_get_dataset_sample(dataset_id):
    """Get sample data from dataset"""
    try:
        user = request.current_user
        num_samples = request.args.get('samples', 3, type=int)
        
        result = custom_dataset_service.get_dataset_sample(dataset_id, user.id, num_samples)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 404 if 'not found' in result['error'] else 400
            
    except Exception as e:
        logging.error(f"Error getting dataset sample: {str(e)}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

@custom_bp.route('/api/datasets/<int:dataset_id>', methods=['DELETE'])
@require_api_key
def api_delete_dataset(dataset_id):
    """Delete a dataset"""
    try:
        user = request.current_user
        result = custom_dataset_service.delete_dataset(dataset_id, user.id)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 404 if 'not found' in result['error'] else 400
            
    except Exception as e:
        logging.error(f"Error deleting dataset: {str(e)}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

# Training API endpoints
@custom_bp.route('/api/models/available', methods=['GET'])
@require_api_key
def api_available_models():
    """List available base models for training"""
    try:
        models = custom_training_service.get_available_base_models()
        return jsonify({
            "success": True,
            "models": models
        })
        
    except Exception as e:
        logging.error(f"Error listing available models: {str(e)}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

@custom_bp.route('/api/training/jobs', methods=['POST'])
@require_api_key
def api_create_training_job():
    """Create a new custom training job"""
    try:
        user = request.current_user
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['job_name', 'base_model', 'dataset_id']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        result = custom_training_service.create_training_job(data, user.id)
        
        if result['success']:
            return jsonify(result), 201
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logging.error(f"Error creating training job: {str(e)}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

@custom_bp.route('/api/training/jobs', methods=['GET'])
@require_api_key
def api_list_training_jobs():
    """List user's training jobs"""
    try:
        user = request.current_user
        jobs = custom_training_service.get_training_jobs(user.id)
        
        job_list = []
        for job in jobs:
            job_dict = {
                'id': job.id,
                'job_name': job.job_name,
                'base_model': job.base_model,
                'dataset_id': job.dataset_id,
                'dataset_name': job.dataset.name if job.dataset else 'Unknown',
                'status': job.status.value,
                'progress': job.progress,
                'epochs': job.epochs,
                'current_epoch': job.current_epoch,
                'current_loss': job.current_loss,
                'best_loss': job.best_loss,
                'created_at': job.created_at.isoformat() if job.created_at else None,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                'use_lora': job.use_lora,
                'learning_rate': job.learning_rate,
                'batch_size': job.batch_size
            }
            job_list.append(job_dict)
        
        return jsonify({
            "success": True,
            "jobs": job_list
        })
        
    except Exception as e:
        logging.error(f"Error listing training jobs: {str(e)}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

@custom_bp.route('/api/training/jobs/<int:job_id>', methods=['GET'])
@require_api_key
def api_get_training_job(job_id):
    """Get training job details"""
    try:
        user = request.current_user
        job = custom_training_service.get_training_job(job_id, user.id)
        
        if not job:
            return jsonify({"success": False, "error": "Training job not found"}), 404
        
        job_dict = {
            'id': job.id,
            'job_name': job.job_name,
            'base_model': job.base_model,
            'dataset_id': job.dataset_id,
            'dataset_name': job.dataset.name if job.dataset else 'Unknown',
            'status': job.status.value,
            'progress': job.progress,
            'epochs': job.epochs,
            'current_epoch': job.current_epoch,
            'current_loss': job.current_loss,
            'best_loss': job.best_loss,
            'training_logs': job.training_logs,
            'error_message': job.error_message,
            'created_at': job.created_at.isoformat() if job.created_at else None,
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'learning_rate': job.learning_rate,
            'batch_size': job.batch_size,
            'max_length': job.max_length,
            'use_lora': job.use_lora,
            'lora_r': job.lora_r,
            'lora_alpha': job.lora_alpha,
            'lora_dropout': job.lora_dropout,
            'use_qlora': job.use_qlora,
            'output_model_name': job.output_model_name,
            'model_id': job.model_id
        }
        
        return jsonify({
            "success": True,
            "job": job_dict
        })
        
    except Exception as e:
        logging.error(f"Error getting training job: {str(e)}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

@custom_bp.route('/api/training/jobs/<int:job_id>/start', methods=['POST'])
@require_api_key
def api_start_training_job(job_id):
    """Start a training job"""
    try:
        user = request.current_user
        result = custom_training_service.start_training_job(job_id, user.id)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logging.error(f"Error starting training job: {str(e)}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

@custom_bp.route('/api/training/jobs/<int:job_id>/stop', methods=['POST'])
@require_api_key
def api_stop_training_job(job_id):
    """Stop a training job"""
    try:
        user = request.current_user
        result = custom_training_service.stop_training_job(job_id, user.id)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logging.error(f"Error stopping training job: {str(e)}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

@custom_bp.route('/api/training/jobs/<int:job_id>/metrics', methods=['GET'])
@require_api_key
def api_get_training_metrics(job_id):
    """Get training metrics for visualization"""
    try:
        user = request.current_user
        result = custom_training_service.get_job_metrics(job_id, user.id)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 404 if 'not found' in result['error'] else 400
            
    except Exception as e:
        logging.error(f"Error getting training metrics: {str(e)}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

@custom_bp.route('/api/training/jobs/<int:job_id>/checkpoints', methods=['GET'])
@require_api_key
def api_get_training_checkpoints(job_id):
    """Get training checkpoints"""
    try:
        user = request.current_user
        checkpoints = custom_training_service.get_job_checkpoints(job_id, user.id)
        
        checkpoint_list = []
        for checkpoint in checkpoints:
            checkpoint_dict = {
                'id': checkpoint.id,
                'checkpoint_name': checkpoint.checkpoint_name,
                'epoch': checkpoint.epoch,
                'step': checkpoint.step,
                'loss_value': checkpoint.loss_value,
                'file_size': checkpoint.file_size,
                'created_at': checkpoint.created_at.isoformat() if checkpoint.created_at else None,
                'perplexity': checkpoint.perplexity,
                'validation_loss': checkpoint.validation_loss
            }
            checkpoint_list.append(checkpoint_dict)
        
        return jsonify({
            "success": True,
            "checkpoints": checkpoint_list
        })
        
    except Exception as e:
        logging.error(f"Error getting training checkpoints: {str(e)}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

@custom_bp.route('/api/training/jobs/<int:job_id>', methods=['DELETE'])
@require_api_key
def api_delete_training_job(job_id):
    """Delete a training job"""
    try:
        user = request.current_user
        result = custom_training_service.delete_training_job(job_id, user.id)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logging.error(f"Error deleting training job: {str(e)}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

def init_custom_routes(app):
    """Initialize custom training routes"""
    app.register_blueprint(custom_bp)
    logging.info("Custom training routes initialized")