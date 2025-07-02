"""
HuggingFace Integration Routes - Model upload and management
"""
import json
import logging
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from huggingface_service import huggingface_service
from models import LLMModel

logger = logging.getLogger(__name__)

# Create blueprint
hf_bp = Blueprint('huggingface', __name__, url_prefix='/huggingface')

@hf_bp.route('/')
def index():
    """HuggingFace integration dashboard"""
    return render_template('huggingface/index.html',
                         is_authenticated=huggingface_service.is_authenticated())

@hf_bp.route('/upload/<int:model_id>')
def upload_form(model_id):
    """Show upload form for a specific model"""
    from app import db
    model = db.session.get(LLMModel, model_id)
    if not model:
        flash('Model not found', 'error')
        return redirect(url_for('huggingface.index'))
    
    return render_template('huggingface/upload.html', 
                         model=model,
                         is_authenticated=huggingface_service.is_authenticated())

@hf_bp.route('/api/upload', methods=['POST'])
def api_upload_model():
    """API endpoint to upload model to HuggingFace"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        model_name = data.get('model_name')
        
        if not model_id or not model_name:
            return jsonify({'success': False, 'error': 'Missing model_id or model_name'}), 400
        
        # Get model from database
        from app import db
        model = db.session.get(LLMModel, model_id)
        if not model:
            return jsonify({'success': False, 'error': 'Model not found'}), 404
        
        # Extract model configuration
        model_config = {
            'temperature': 0.7,
            'max_length': 100,
            'model_type': model.base_model,
            'description': model.description or 'Fine-tuned model from LLM Platform'
        }
        
        # Upload to HuggingFace
        result = huggingface_service.upload_model(model_id, model_name, model_config)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@hf_bp.route('/api/models')
def api_list_models():
    """List user's models on HuggingFace"""
    try:
        models = huggingface_service.list_user_models()
        return jsonify({'models': models})
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({'error': str(e)}), 500

@hf_bp.route('/api/model/<path:repo_name>')
def api_get_model_info(repo_name):
    """Get information about a specific model"""
    try:
        info = huggingface_service.get_model_info(repo_name)
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500

@hf_bp.route('/api/model/<path:repo_name>', methods=['DELETE'])
def api_delete_model(repo_name):
    """Delete a model from HuggingFace"""
    try:
        result = huggingface_service.delete_model(repo_name)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        return jsonify({'error': str(e)}), 500

@hf_bp.route('/auth')
def auth_status():
    """Check authentication status"""
    return jsonify({
        'authenticated': huggingface_service.is_authenticated(),
        'username': huggingface_service.username if huggingface_service.is_authenticated() else None
    })

def init_huggingface_routes(app):
    """Initialize HuggingFace routes"""
    app.register_blueprint(hf_bp)