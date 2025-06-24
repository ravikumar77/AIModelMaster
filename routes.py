from flask import render_template, request, redirect, url_for, flash, jsonify
from app import app, db
from models import LLMModel, TrainingJob, Evaluation, GenerationLog, ModelStatus, TrainingStatus
from llm_service import LLMService
from training_service import TrainingService
import json
from datetime import datetime

llm_service = LLMService()
training_service = TrainingService()

@app.route('/')
def index():
    # Get recent activity
    recent_models = LLMModel.query.order_by(LLMModel.updated_at.desc()).limit(5).all()
    recent_jobs = TrainingJob.query.order_by(TrainingJob.created_at.desc()).limit(5).all()
    recent_generations = GenerationLog.query.order_by(GenerationLog.created_at.desc()).limit(5).all()
    
    # Get statistics
    total_models = LLMModel.query.count()
    active_jobs = TrainingJob.query.filter(TrainingJob.status.in_([TrainingStatus.RUNNING, TrainingStatus.PENDING])).count()
    total_generations = GenerationLog.query.count()
    
    return render_template('index.html', 
                         recent_models=recent_models,
                         recent_jobs=recent_jobs,
                         recent_generations=recent_generations,
                         total_models=total_models,
                         active_jobs=active_jobs,
                         total_generations=total_generations)

@app.route('/models')
def models():
    page = request.args.get('page', 1, type=int)
    models_query = LLMModel.query.order_by(LLMModel.created_at.desc())
    models_paginated = models_query.paginate(page=page, per_page=10, error_out=False)
    return render_template('models.html', models=models_paginated)

@app.route('/models/new', methods=['GET', 'POST'])
def new_model():
    if request.method == 'POST':
        name = request.form['name']
        base_model = request.form['base_model']
        description = request.form.get('description', '')
        
        model = LLMModel(
            name=name,
            base_model=base_model,
            description=description,
            model_size=llm_service.get_model_size(base_model)
        )
        
        db.session.add(model)
        db.session.commit()
        
        flash(f'Model "{name}" created successfully!', 'success')
        return redirect(url_for('models'))
    
    available_models = llm_service.get_available_models()
    return render_template('models.html', available_models=available_models, show_form=True)

@app.route('/models/<int:model_id>')
def model_detail(model_id):
    model = LLMModel.query.get_or_404(model_id)
    training_jobs = TrainingJob.query.filter_by(model_id=model_id).order_by(TrainingJob.created_at.desc()).all()
    evaluations = Evaluation.query.filter_by(model_id=model_id).order_by(Evaluation.created_at.desc()).all()
    return render_template('models.html', model=model, training_jobs=training_jobs, evaluations=evaluations, show_detail=True)

@app.route('/training')
def training():
    page = request.args.get('page', 1, type=int)
    jobs_query = TrainingJob.query.order_by(TrainingJob.created_at.desc())
    jobs_paginated = jobs_query.paginate(page=page, per_page=10, error_out=False)
    return render_template('training.html', jobs=jobs_paginated)

@app.route('/training/new', methods=['GET', 'POST'])
def new_training():
    if request.method == 'POST':
        model_id = request.form['model_id']
        job_name = request.form['job_name']
        epochs = int(request.form.get('epochs', 3))
        learning_rate = float(request.form.get('learning_rate', 0.0001))
        batch_size = int(request.form.get('batch_size', 8))
        lora_r = int(request.form.get('lora_r', 8))
        lora_alpha = int(request.form.get('lora_alpha', 32))
        lora_dropout = float(request.form.get('lora_dropout', 0.05))
        
        job = TrainingJob(
            model_id=model_id,
            job_name=job_name,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        db.session.add(job)
        db.session.commit()
        
        # Start training simulation
        training_service.start_training_simulation(job.id)
        
        flash(f'Training job "{job_name}" started successfully!', 'success')
        return redirect(url_for('training'))
    
    models = LLMModel.query.filter_by(status=ModelStatus.AVAILABLE).all()
    return render_template('training.html', models=models, show_form=True)

@app.route('/training/<int:job_id>')
def training_detail(job_id):
    job = TrainingJob.query.get_or_404(job_id)
    return render_template('training.html', job=job, show_detail=True)

@app.route('/inference')
def inference():
    models = LLMModel.query.filter_by(status=ModelStatus.AVAILABLE).all()
    recent_generations = GenerationLog.query.order_by(GenerationLog.created_at.desc()).limit(10).all()
    return render_template('inference.html', models=models, recent_generations=recent_generations)

@app.route('/inference/generate', methods=['POST'])
def generate_text():
    model_id = request.form['model_id']
    prompt = request.form['prompt']
    temperature = float(request.form.get('temperature', 0.7))
    max_length = int(request.form.get('max_length', 100))
    top_p = float(request.form.get('top_p', 0.9))
    top_k = int(request.form.get('top_k', 50))
    
    try:
        start_time = datetime.now()
        response = llm_service.generate_text(
            model_id=model_id,
            prompt=prompt,
            temperature=temperature,
            max_length=max_length,
            top_p=top_p,
            top_k=top_k
        )
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # Log the generation
        log = GenerationLog(
            model_id=model_id,
            prompt=prompt,
            response=response,
            temperature=temperature,
            max_length=max_length,
            top_p=top_p,
            top_k=top_k,
            generation_time=generation_time
        )
        db.session.add(log)
        db.session.commit()
        
        flash('Text generated successfully!', 'success')
        return redirect(url_for('inference'))
        
    except Exception as e:
        flash(f'Generation failed: {str(e)}', 'error')
        return redirect(url_for('inference'))

@app.route('/evaluation')
def evaluation():
    models = LLMModel.query.filter_by(status=ModelStatus.AVAILABLE).all()
    evaluations = Evaluation.query.order_by(Evaluation.created_at.desc()).limit(10).all()
    return render_template('evaluation.html', models=models, evaluations=evaluations)

@app.route('/evaluation/run', methods=['POST'])
def run_evaluation():
    model_id = request.form['model_id']
    eval_name = request.form['eval_name']
    
    try:
        # Run mock evaluation
        metrics = training_service.run_evaluation(model_id)
        
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
        
        flash(f'Evaluation "{eval_name}" completed successfully!', 'success')
        
    except Exception as e:
        flash(f'Evaluation failed: {str(e)}', 'error')
    
    return redirect(url_for('evaluation'))

@app.route('/export')
def export():
    models = LLMModel.query.filter_by(status=ModelStatus.AVAILABLE).all()
    return render_template('export.html', models=models)

@app.route('/export/onnx', methods=['POST'])
def export_onnx():
    model_id = request.form['model_id']
    
    try:
        # Simulate ONNX export
        model = LLMModel.query.get_or_404(model_id)
        model.status = ModelStatus.EXPORTING
        db.session.commit()
        
        # Simulate export process
        training_service.simulate_export(model_id)
        
        model.status = ModelStatus.AVAILABLE
        db.session.commit()
        
        flash(f'Model "{model.name}" exported to ONNX successfully!', 'success')
        
    except Exception as e:
        model = LLMModel.query.get_or_404(model_id)
        model.status = ModelStatus.ERROR
        db.session.commit()
        flash(f'Export failed: {str(e)}', 'error')
    
    return redirect(url_for('export'))

# API endpoints for AJAX calls (for charts and real-time updates)
@app.route('/api/training/<int:job_id>/progress')
def get_training_progress(job_id):
    job = TrainingJob.query.get_or_404(job_id)
    return jsonify({
        'progress': job.progress,
        'status': job.status.value,
        'current_epoch': job.current_epoch,
        'current_loss': job.current_loss,
        'logs': job.logs
    })

@app.route('/api/models/<int:model_id>/metrics')
def get_model_metrics(model_id):
    evaluations = Evaluation.query.filter_by(model_id=model_id).all()
    metrics = []
    for eval in evaluations:
        metrics.append({
            'name': eval.eval_name,
            'perplexity': eval.perplexity,
            'bleu_score': eval.bleu_score,
            'rouge_score': eval.rouge_score,
            'created_at': eval.created_at.isoformat()
        })
    return jsonify(metrics)
