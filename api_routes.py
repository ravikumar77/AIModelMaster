"""
API routes for LLM platform - RESTful endpoints for external access
"""
from flask import jsonify, request, g
from app import app, db
from models import LLMModel, TrainingJob, Evaluation, GenerationLog, ModelStatus, TrainingStatus, User, APIKey, CodingDataset
from llm_service import LLMService
from training_service import TrainingService
from auth_service import auth_service, require_api_key
from coding_training import coding_service
from datetime import datetime
import logging

# Initialize services
llm_service = LLMService()
training_service = TrainingService()

# API Error handlers
@app.errorhandler(404)
def api_not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Resource not found'}), 404
    return error

@app.errorhandler(400)
def api_bad_request(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Bad request'}), 400
    return error

@app.errorhandler(500)
def api_internal_error(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return error

# API Routes

@app.route('/api/health', methods=['GET'])
def api_health():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/models', methods=['GET'])
def api_list_models():
    """List all available models"""
    try:
        models = LLMModel.query.all()
        return jsonify({
            'models': [{
                'id': model.id,
                'name': model.name,
                'base_model': model.base_model,
                'status': model.status.value,
                'description': model.description,
                'model_size': model.model_size,
                'created_at': model.created_at.isoformat() if model.created_at else None,
                'updated_at': model.updated_at.isoformat() if model.updated_at else None
            } for model in models]
        })
    except Exception as e:
        logging.error(f"API error listing models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<int:model_id>', methods=['GET'])
def api_get_model(model_id):
    """Get specific model information"""
    try:
        model = LLMModel.query.get_or_404(model_id)
        return jsonify({
            'id': model.id,
            'name': model.name,
            'base_model': model.base_model,
            'status': model.status.value,
            'description': model.description,
            'model_size': model.model_size,
            'created_at': model.created_at.isoformat() if model.created_at else None,
            'updated_at': model.updated_at.isoformat() if model.updated_at else None,
            'parameters': model.parameters
        })
    except Exception as e:
        logging.error(f"API error getting model {model_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['POST'])
def api_create_model():
    """Create a new model"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON data required'}), 400
            
        required_fields = ['name', 'base_model']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        model = LLMModel(
            name=data['name'],
            base_model=data['base_model'],
            description=data.get('description', ''),
            model_size=data.get('model_size', ''),
            status=ModelStatus.AVAILABLE
        )
        
        db.session.add(model)
        db.session.commit()
        
        return jsonify({
            'id': model.id,
            'name': model.name,
            'base_model': model.base_model,
            'status': model.status.value,
            'message': 'Model created successfully'
        }), 201
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"API error creating model: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<int:model_id>/generate', methods=['POST'])
@require_api_key
def api_generate_text(model_id):
    """Generate text using a specific model"""
    try:
        model = LLMModel.query.get_or_404(model_id)
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Prompt is required'}), 400
        
        prompt = data['prompt']
        temperature = data.get('temperature', 0.7)
        max_length = data.get('max_length', 100)
        top_p = data.get('top_p', 0.9)
        top_k = data.get('top_k', 50)
        
        # Generate text using LLM service
        start_time = datetime.utcnow()
        result = llm_service.generate_text(
            model_id=model_id,
            prompt=prompt,
            temperature=temperature,
            max_length=max_length,
            top_p=top_p,
            top_k=top_k
        )
        generation_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Log generation
        log_entry = GenerationLog(
            model_id=model_id,
            prompt=prompt,
            response=result['text'],
            temperature=temperature,
            max_length=max_length,
            top_p=top_p,
            top_k=top_k,
            generation_time=generation_time,
            api_key_id=g.api_key.id if hasattr(g, 'api_key') else None
        )
        db.session.add(log_entry)
        db.session.commit()
        
        return jsonify({
            'text': result['text'],
            'model_id': model_id,
            'model_name': model.name,
            'generation_time': generation_time,
            'parameters': {
                'temperature': temperature,
                'max_length': max_length,
                'top_p': top_p,
                'top_k': top_k
            },
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logging.error(f"API error generating text: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training', methods=['GET'])
def api_list_training_jobs():
    """List training jobs"""
    try:
        jobs = TrainingJob.query.order_by(TrainingJob.created_at.desc()).all()
        return jsonify({
            'training_jobs': [{
                'id': job.id,
                'model_id': job.model_id,
                'job_name': job.job_name,
                'status': job.status.value,
                'progress': job.progress,
                'epochs': job.epochs,
                'current_epoch': job.current_epoch,
                'learning_rate': job.learning_rate,
                'batch_size': job.batch_size,
                'created_at': job.created_at.isoformat() if job.created_at else None,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None
            } for job in jobs]
        })
    except Exception as e:
        logging.error(f"API error listing training jobs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training', methods=['POST'])
def api_create_training_job():
    """Create a new training job"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON data required'}), 400
            
        required_fields = ['model_id', 'job_name']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate model exists
        model = LLMModel.query.get_or_404(data['model_id'])
        
        job = TrainingJob(
            model_id=data['model_id'],
            job_name=data['job_name'],
            epochs=data.get('epochs', 3),
            learning_rate=data.get('learning_rate', 0.0001),
            batch_size=data.get('batch_size', 8),
            lora_r=data.get('lora_r', 8),
            lora_alpha=data.get('lora_alpha', 32),
            lora_dropout=data.get('lora_dropout', 0.05),
            status=TrainingStatus.PENDING
        )
        
        db.session.add(job)
        db.session.commit()
        
        # Start training simulation
        training_service.start_training_simulation(job.id)
        
        return jsonify({
            'id': job.id,
            'job_name': job.job_name,
            'model_id': job.model_id,
            'status': job.status.value,
            'message': 'Training job created and started successfully'
        }), 201
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"API error creating training job: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/<int:job_id>', methods=['GET'])
def api_get_training_job(job_id):
    """Get training job details"""
    try:
        job = TrainingJob.query.get_or_404(job_id)
        return jsonify({
            'id': job.id,
            'model_id': job.model_id,
            'job_name': job.job_name,
            'status': job.status.value,
            'progress': job.progress,
            'epochs': job.epochs,
            'current_epoch': job.current_epoch,
            'learning_rate': job.learning_rate,
            'batch_size': job.batch_size,
            'lora_r': job.lora_r,
            'lora_alpha': job.lora_alpha,
            'lora_dropout': job.lora_dropout,
            'current_loss': job.current_loss,
            'created_at': job.created_at.isoformat() if job.created_at else None,
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'logs': job.logs
        })
    except Exception as e:
        logging.error(f"API error getting training job {job_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/<int:job_id>/pause', methods=['POST'])
def api_pause_training(job_id):
    """Pause a training job"""
    try:
        job = TrainingJob.query.get_or_404(job_id)
        
        if job.status != TrainingStatus.RUNNING:
            return jsonify({'error': 'Job is not running'}), 400
        
        training_service.pause_training(job_id)
        
        return jsonify({
            'id': job_id,
            'status': 'PAUSED',
            'message': 'Training job paused successfully'
        })
        
    except Exception as e:
        logging.error(f"API error pausing training job {job_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/<int:job_id>/resume', methods=['POST'])
def api_resume_training(job_id):
    """Resume a paused training job"""
    try:
        job = TrainingJob.query.get_or_404(job_id)
        
        if job.status != TrainingStatus.PAUSED:
            return jsonify({'error': 'Job is not paused'}), 400
        
        training_service.resume_training(job_id)
        
        return jsonify({
            'id': job_id,
            'status': 'RUNNING',
            'message': 'Training job resumed successfully'
        })
        
    except Exception as e:
        logging.error(f"API error resuming training job {job_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluations', methods=['GET'])
def api_list_evaluations():
    """List model evaluations"""
    try:
        evaluations = Evaluation.query.order_by(Evaluation.created_at.desc()).all()
        return jsonify({
            'evaluations': [{
                'id': eval.id,
                'model_id': eval.model_id,
                'eval_name': eval.eval_name,
                'perplexity': eval.perplexity,
                'bleu_score': eval.bleu_score,
                'rouge_score': eval.rouge_score,
                'response_diversity': eval.response_diversity,
                'avg_response_length': eval.avg_response_length,
                'created_at': eval.created_at.isoformat() if eval.created_at else None
            } for eval in evaluations]
        })
    except Exception as e:
        logging.error(f"API error listing evaluations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<int:model_id>/evaluate', methods=['POST'])
def api_evaluate_model(model_id):
    """Run evaluation on a model"""
    try:
        model = LLMModel.query.get_or_404(model_id)
        data = request.get_json() or {}
        
        eval_name = data.get('eval_name', f'Evaluation {datetime.utcnow().strftime("%Y-%m-%d %H:%M")}')
        
        # Run evaluation using training service
        metrics = training_service.run_evaluation(model_id)
        
        # Create evaluation record
        evaluation = Evaluation(
            model_id=model_id,
            eval_name=eval_name,
            perplexity=metrics['perplexity'],
            bleu_score=metrics['bleu_score'],
            rouge_score=metrics['rouge_score'],
            response_diversity=metrics['response_diversity'],
            avg_response_length=metrics['avg_response_length']
        )
        
        db.session.add(evaluation)
        db.session.commit()
        
        return jsonify({
            'id': evaluation.id,
            'model_id': model_id,
            'eval_name': eval_name,
            'metrics': metrics,
            'message': 'Evaluation completed successfully'
        }), 201
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"API error evaluating model {model_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def api_get_statistics():
    """Get platform statistics"""
    try:
        total_models = LLMModel.query.count()
        total_training_jobs = TrainingJob.query.count()
        running_jobs = TrainingJob.query.filter_by(status=TrainingStatus.RUNNING).count()
        completed_jobs = TrainingJob.query.filter_by(status=TrainingStatus.COMPLETED).count()
        total_evaluations = Evaluation.query.count()
        total_generations = GenerationLog.query.count()
        
        return jsonify({
            'total_models': total_models,
            'total_training_jobs': total_training_jobs,
            'running_training_jobs': running_jobs,
            'completed_training_jobs': completed_jobs,
            'total_evaluations': total_evaluations,
            'total_generations': total_generations,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logging.error(f"API error getting statistics: {e}")
        return jsonify({'error': str(e)}), 500

# User and API Key Management

@app.route('/api/auth/register', methods=['POST'])
def api_register():
    """Register a new user and get API key"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON data required'}), 400
            
        required_fields = ['username', 'email', 'password']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        user, result = auth_service.create_user(
            data['username'], 
            data['email'], 
            data['password']
        )
        
        if not user:
            return jsonify({'error': result}), 400
        
        return jsonify({
            'user_id': user.id,
            'username': user.username,
            'api_key': result.key_value,
            'message': 'User registered successfully'
        }), 201
        
    except Exception as e:
        logging.error(f"API error registering user: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/keys', methods=['GET'])
@require_api_key
def api_list_keys():
    """List user's API keys"""
    try:
        keys = APIKey.query.filter_by(user_id=g.user_id, is_active=True).all()
        return jsonify({
            'api_keys': [{
                'id': key.id,
                'key_name': key.key_name,
                'created_at': key.created_at.isoformat() if key.created_at else None,
                'last_used': key.last_used.isoformat() if key.last_used else None,
                'usage_count': key.usage_count,
                'rate_limit': key.rate_limit
            } for key in keys]
        })
    except Exception as e:
        logging.error(f"API error listing keys: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/keys', methods=['POST'])
@require_api_key
def api_create_key():
    """Create a new API key"""
    try:
        data = request.get_json() or {}
        key_name = data.get('key_name', 'New API Key')
        rate_limit = data.get('rate_limit', 1000)
        
        api_key = auth_service.create_api_key(g.user_id, key_name, rate_limit)
        
        if not api_key:
            return jsonify({'error': 'Failed to create API key'}), 500
        
        return jsonify({
            'id': api_key.id,
            'key_name': api_key.key_name,
            'key_value': api_key.key_value,
            'rate_limit': api_key.rate_limit,
            'message': 'API key created successfully'
        }), 201
        
    except Exception as e:
        logging.error(f"API error creating key: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/keys/<int:key_id>', methods=['DELETE'])
@require_api_key
def api_delete_key(key_id):
    """Deactivate an API key"""
    try:
        success = auth_service.deactivate_api_key(key_id, g.user_id)
        
        if success:
            return jsonify({'message': 'API key deactivated successfully'})
        else:
            return jsonify({'error': 'API key not found'}), 404
            
    except Exception as e:
        logging.error(f"API error deleting key: {e}")
        return jsonify({'error': str(e)}), 500

# Coding Datasets and Training

@app.route('/api/datasets', methods=['GET'])
def api_list_datasets():
    """List coding datasets"""
    try:
        datasets = coding_service.get_datasets()
        return jsonify({
            'datasets': [{
                'id': dataset.id,
                'name': dataset.name,
                'description': dataset.description,
                'language': dataset.language,
                'dataset_type': dataset.dataset_type,
                'created_at': dataset.created_at.isoformat() if dataset.created_at else None
            } for dataset in datasets]
        })
    except Exception as e:
        logging.error(f"API error listing datasets: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/coding', methods=['POST'])
@require_api_key
def api_create_coding_training():
    """Create a coding-specific training job"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON data required'}), 400
            
        required_fields = ['model_id', 'job_name', 'dataset_id']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate model and dataset exist
        model = LLMModel.query.get_or_404(data['model_id'])
        dataset = CodingDataset.query.get_or_404(data['dataset_id'])
        
        job = TrainingJob(
            model_id=data['model_id'],
            job_name=data['job_name'],
            dataset_id=data['dataset_id'],
            training_type='coding',
            epochs=data.get('epochs', 5),
            learning_rate=data.get('learning_rate', 0.00005),  # Lower for coding
            batch_size=data.get('batch_size', 4),
            lora_r=data.get('lora_r', 16),  # Higher for coding
            lora_alpha=data.get('lora_alpha', 32),
            lora_dropout=data.get('lora_dropout', 0.1),
            status=TrainingStatus.PENDING
        )
        
        db.session.add(job)
        db.session.commit()
        
        # Start training simulation
        training_service.start_training_simulation(job.id)
        
        return jsonify({
            'id': job.id,
            'job_name': job.job_name,
            'model_id': job.model_id,
            'dataset_id': job.dataset_id,
            'training_type': job.training_type,
            'status': job.status.value,
            'message': 'Coding training job created and started successfully'
        }), 201
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"API error creating coding training job: {e}")
        return jsonify({'error': str(e)}), 500