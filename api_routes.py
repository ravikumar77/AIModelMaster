"""
API routes for LLM platform - RESTful endpoints for external access
"""
from flask import jsonify, request
from app import app, db
from models import LLMModel, TrainingJob, Evaluation, GenerationLog, ModelStatus, TrainingStatus
from llm_service import LLMService
from training_service import TrainingService
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
            generation_time=generation_time
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