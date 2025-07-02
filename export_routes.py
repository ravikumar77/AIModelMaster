"""
Export Routes - Advanced model export capabilities
Handles Triton, TensorFlow Lite, and enhanced HuggingFace Hub exports
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask import current_app
import json
import logging
from datetime import datetime
from models import LLMModel, ExportJob, ExportStatus
from export_service import ExportService
from auth_service import require_api_key

logger = logging.getLogger(__name__)

# Create Blueprint
export_bp = Blueprint('export', __name__, url_prefix='/export')

# Initialize export service
export_service = ExportService()

@export_bp.route('/')
def index():
    """Export dashboard - show available models and recent export jobs"""
    try:
        # Get available models for export
        models = LLMModel.query.filter_by(status='AVAILABLE').all()
        
        # Get recent export jobs
        recent_jobs = ExportJob.query.order_by(ExportJob.created_at.desc()).limit(10).all()
        
        return render_template('export/index.html', 
                             models=models, 
                             recent_jobs=recent_jobs)
                             
    except Exception as e:
        logger.error(f"Error loading export dashboard: {e}")
        flash('Error loading export dashboard', 'error')
        return redirect(url_for('index'))

@export_bp.route('/model/<int:model_id>')
def model_export_options(model_id):
    """Show export options for a specific model"""
    try:
        model = LLMModel.query.get_or_404(model_id)
        
        # Get export history for this model
        export_jobs = ExportJob.query.filter_by(model_id=model_id)\
                                   .order_by(ExportJob.created_at.desc())\
                                   .limit(20).all()
        
        return render_template('export/model_options.html',
                             model=model,
                             export_jobs=export_jobs)
                             
    except Exception as e:
        logger.error(f"Error loading model export options: {e}")
        flash('Error loading model export options', 'error')
        return redirect(url_for('export.index'))

# API Endpoints for Export Operations

@export_bp.route('/api/<int:model_id>/triton', methods=['POST'])
@require_api_key
def api_export_triton(model_id):
    """API endpoint to export model for Triton Inference Server"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON data required'}), 400
        
        # Validate model exists and is available
        model = LLMModel.query.get_or_404(model_id)
        if model.status != 'AVAILABLE':
            return jsonify({'error': 'Model not available for export'}), 400
        
        # Extract configuration with defaults
        config = {
            'batch_size': data.get('batch_size', 1),
            'max_sequence_length': data.get('max_sequence_length', 512),
            'dynamic_shape': data.get('dynamic_shape', False),
            'optimization_level': data.get('optimization_level', 'basic'),
            'precision': data.get('precision', 'fp32')
        }
        
        # Validate configuration
        if config['batch_size'] < 1 or config['batch_size'] > 32:
            return jsonify({'error': 'Batch size must be between 1 and 32'}), 400
            
        if config['max_sequence_length'] < 64 or config['max_sequence_length'] > 2048:
            return jsonify({'error': 'Max sequence length must be between 64 and 2048'}), 400
        
        # Create export job
        job = export_service.export_to_triton(model_id, config)
        
        return jsonify({
            'job_id': job.id,
            'status': job.status.value,
            'message': 'Triton export job created successfully',
            'config': config
        }), 202
        
    except Exception as e:
        logger.error(f"Error creating Triton export job: {e}")
        return jsonify({'error': 'Failed to create export job'}), 500

@export_bp.route('/api/<int:model_id>/tflite', methods=['POST'])
@require_api_key
def api_export_tflite(model_id):
    """API endpoint to export model to TensorFlow Lite"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON data required'}), 400
        
        # Validate model exists and is available
        model = LLMModel.query.get_or_404(model_id)
        if model.status != 'AVAILABLE':
            return jsonify({'error': 'Model not available for export'}), 400
        
        # Extract configuration with defaults
        config = {
            'quantization': data.get('quantization', 'none'),  # none, dynamic, float16, int8
            'target_device': data.get('target_device', 'generic'),  # generic, android, edgetpu
            'max_sequence_length': data.get('max_sequence_length', 256),
            'optimize_for_size': data.get('optimize_for_size', True),
            'enable_select_tf_ops': data.get('enable_select_tf_ops', False)
        }
        
        # Validate configuration
        valid_quantizations = ['none', 'dynamic', 'float16', 'int8']
        if config['quantization'] not in valid_quantizations:
            return jsonify({'error': f'Invalid quantization. Must be one of: {valid_quantizations}'}), 400
            
        valid_devices = ['generic', 'android', 'edgetpu', 'ios']
        if config['target_device'] not in valid_devices:
            return jsonify({'error': f'Invalid target device. Must be one of: {valid_devices}'}), 400
        
        # Create export job
        job = export_service.export_to_tflite(model_id, config)
        
        return jsonify({
            'job_id': job.id,
            'status': job.status.value,
            'message': 'TensorFlow Lite export job created successfully',
            'config': config
        }), 202
        
    except Exception as e:
        logger.error(f"Error creating TFLite export job: {e}")
        return jsonify({'error': 'Failed to create export job'}), 500

@export_bp.route('/api/<int:model_id>/huggingface', methods=['POST'])
@require_api_key
def api_export_huggingface(model_id):
    """API endpoint to export model to HuggingFace Hub"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON data required'}), 400
        
        # Validate model exists and is available
        model = LLMModel.query.get_or_404(model_id)
        if model.status != 'AVAILABLE':
            return jsonify({'error': 'Model not available for export'}), 400
        
        # Extract required configuration
        required_fields = ['repo_name', 'hf_token']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        config = {
            'repo_name': data['repo_name'],
            'hf_token': data['hf_token'],
            'visibility': data.get('visibility', 'private'),  # private, public
            'auto_readme': data.get('auto_readme', True),
            'model_description': data.get('description', ''),
            'tags': data.get('tags', []),
            'license': data.get('license', 'apache-2.0')
        }
        
        # Validate repository name format
        import re
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9-_.]{0,95}$', config['repo_name']):
            return jsonify({'error': 'Invalid repository name format'}), 400
        
        # Create export job
        job = export_service.export_to_huggingface(model_id, config)
        
        return jsonify({
            'job_id': job.id,
            'status': job.status.value,
            'message': 'HuggingFace Hub export job created successfully',
            'repo_name': config['repo_name'],
            'visibility': config['visibility']
        }), 202
        
    except Exception as e:
        logger.error(f"Error creating HuggingFace export job: {e}")
        return jsonify({'error': 'Failed to create export job'}), 500

@export_bp.route('/api/jobs/<int:job_id>', methods=['GET'])
@require_api_key
def api_get_export_job(job_id):
    """Get export job status and details"""
    try:
        job = export_service.get_export_job(job_id)
        if not job:
            return jsonify({'error': 'Export job not found'}), 404
        
        # Parse config if available
        config = {}
        if job.config:
            try:
                config = json.loads(job.config)
            except json.JSONDecodeError:
                pass
        
        response_data = {
            'id': job.id,
            'model_id': job.model_id,
            'export_type': job.export_type,
            'status': job.status.value,
            'config': config,
            'created_at': job.created_at.isoformat() if job.created_at else None,
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'logs': job.logs,
            'error_message': job.error_message
        }
        
        # Add output information if completed
        if job.status == ExportStatus.COMPLETED and job.output_path:
            if job.export_type == 'huggingface':
                response_data['output_url'] = job.output_path
            else:
                response_data['download_available'] = True
                response_data['download_url'] = url_for('export.download_export', job_id=job.id, _external=True)
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error getting export job {job_id}: {e}")
        return jsonify({'error': 'Failed to get export job'}), 500

@export_bp.route('/api/jobs', methods=['GET'])
@require_api_key
def api_list_export_jobs():
    """List export jobs with optional filtering"""
    try:
        model_id = request.args.get('model_id', type=int)
        export_type = request.args.get('export_type')
        status = request.args.get('status')
        limit = request.args.get('limit', default=50, type=int)
        
        query = ExportJob.query
        
        if model_id:
            query = query.filter_by(model_id=model_id)
        if export_type:
            query = query.filter_by(export_type=export_type)
        if status:
            query = query.filter_by(status=ExportStatus(status))
        
        jobs = query.order_by(ExportJob.created_at.desc()).limit(limit).all()
        
        jobs_data = []
        for job in jobs:
            config = {}
            if job.config:
                try:
                    config = json.loads(job.config)
                except json.JSONDecodeError:
                    pass
            
            jobs_data.append({
                'id': job.id,
                'model_id': job.model_id,
                'export_type': job.export_type,
                'status': job.status.value,
                'created_at': job.created_at.isoformat() if job.created_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                'config': config
            })
        
        return jsonify({
            'jobs': jobs_data,
            'total': len(jobs_data)
        })
        
    except Exception as e:
        logger.error(f"Error listing export jobs: {e}")
        return jsonify({'error': 'Failed to list export jobs'}), 500

@export_bp.route('/download/<int:job_id>')
def download_export(job_id):
    """Download exported model files"""
    try:
        job = export_service.get_export_job(job_id)
        if not job:
            flash('Export job not found', 'error')
            return redirect(url_for('export.index'))
        
        if job.status != ExportStatus.COMPLETED:
            flash('Export job not completed yet', 'warning')
            return redirect(url_for('export.index'))
        
        if not job.output_path:
            flash('No output file available for download', 'error')
            return redirect(url_for('export.index'))
        
        # For now, show download information
        # TODO: Implement actual file serving
        return render_template('export/download.html', job=job)
        
    except Exception as e:
        logger.error(f"Error downloading export {job_id}: {e}")
        flash('Error downloading export', 'error')
        return redirect(url_for('export.index'))

def init_export_routes(app):
    """Initialize export routes"""
    app.register_blueprint(export_bp)
    logger.info("Export routes initialized")