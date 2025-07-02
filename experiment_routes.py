"""
Experiment Tracking & Comparison Routes - Flask web interface for ML experiments
"""
import json
import logging
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, session
from models import Experiment, ExperimentMetric, ExperimentComparison, LLMModel, CodingDataset

logger = logging.getLogger(__name__)

# Create blueprint
experiment_bp = Blueprint('experiments', __name__, url_prefix='/experiments')

# Global service instance (will be initialized later to avoid circular imports)
experiment_tracking_service = None

def get_experiment_service():
    """Get experiment tracking service instance (lazy initialization)"""
    global experiment_tracking_service
    if experiment_tracking_service is None:
        from experiment_tracking_service import ExperimentTrackingService
        experiment_tracking_service = ExperimentTrackingService()
    return experiment_tracking_service

@experiment_bp.route('/')
def index():
    """Experiment tracking dashboard"""
    try:
        # Get filter parameters
        status_filter = request.args.get('status')
        tag_filter = request.args.get('tag')
        group_filter = request.args.get('group')
        user_id = session.get('user_id')
        
        # Get experiments with filtering
        tags = [tag_filter] if tag_filter else None
        experiments = get_experiment_service().get_experiments(
            user_id=user_id,
            status=status_filter,
            experiment_group=group_filter,
            tags=tags,
            limit=50
        )
        
        # Get statistics
        stats = get_experiment_service().get_experiment_statistics()
        
        # Get available models and datasets for creating new experiments
        models = LLMModel.query.all()
        datasets = CodingDataset.query.all()
        
        # Get unique tags and groups for filtering
        all_experiments = Experiment.query.all()
        all_tags = set()
        all_groups = set()
        
        for exp in all_experiments:
            if exp.tags:
                try:
                    tags_list = json.loads(exp.tags)
                    all_tags.update(tags_list)
                except json.JSONDecodeError:
                    pass
            if exp.experiment_group:
                all_groups.add(exp.experiment_group)
        
        return render_template('experiments/index.html',
                             experiments=experiments,
                             stats=stats,
                             models=models,
                             datasets=datasets,
                             all_tags=sorted(all_tags),
                             all_groups=sorted(all_groups),
                             current_status=status_filter,
                             current_tag=tag_filter,
                             current_group=group_filter)
                             
    except Exception as e:
        logger.error(f"Error loading experiments dashboard: {e}")
        flash(f"Error loading experiments: {str(e)}", 'error')
        return render_template('experiments/index.html',
                             experiments=[], stats={}, models=[], datasets=[],
                             all_tags=[], all_groups=[])

@experiment_bp.route('/create', methods=['GET', 'POST'])
def create():
    """Create a new experiment"""
    if request.method == 'GET':
        models = LLMModel.query.all()
        datasets = CodingDataset.query.all()
        return render_template('experiments/create.html', models=models, datasets=datasets)
    
    try:
        data = request.get_json() if request.is_json else request.form
        
        # Extract form data
        name = data.get('name')
        description = data.get('description', '')
        model_id = int(data.get('model_id'))
        dataset_id = data.get('dataset_id')
        dataset_id = int(dataset_id) if dataset_id else None
        
        # Parse hyperparameters
        hyperparameters = {
            'learning_rate': float(data.get('learning_rate', 0.0001)),
            'batch_size': int(data.get('batch_size', 8)),
            'epochs': int(data.get('epochs', 20)),
            'lora_r': int(data.get('lora_r', 8)),
            'lora_alpha': int(data.get('lora_alpha', 32)),
            'lora_dropout': float(data.get('lora_dropout', 0.05)),
        }
        
        # Parse runtime settings
        runtime_settings = {
            'gpu_type': data.get('gpu_type', 'Tesla T4'),
            'precision': data.get('precision', 'fp16'),
            'gradient_checkpointing': data.get('gradient_checkpointing', 'true') == 'true',
            'max_grad_norm': float(data.get('max_grad_norm', 1.0))
        }
        
        # Parse tags
        tags_str = data.get('tags', '')
        tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
        
        experiment_group = data.get('experiment_group', '')
        user_id = session.get('user_id')
        
        # Create experiment
        experiment = get_experiment_service().create_experiment(
            name=name,
            description=description,
            model_id=model_id,
            hyperparameters=hyperparameters,
            dataset_id=dataset_id,
            runtime_settings=runtime_settings,
            created_by=user_id,
            tags=tags,
            experiment_group=experiment_group or None
        )
        
        if request.is_json:
            return jsonify({
                'success': True,
                'experiment_id': experiment.id,
                'message': 'Experiment created successfully'
            })
        else:
            flash('Experiment created successfully!', 'success')
            return redirect(url_for('experiments.detail', experiment_id=experiment.id))
            
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        error_msg = f"Error creating experiment: {str(e)}"
        
        if request.is_json:
            return jsonify({'success': False, 'error': error_msg}), 400
        else:
            flash(error_msg, 'error')
            return redirect(url_for('experiments.create'))

@experiment_bp.route('/<int:experiment_id>')
def detail(experiment_id):
    """Experiment detail view with metrics and analysis"""
    try:
        experiment = Experiment.query.get_or_404(experiment_id)
        
        # Get experiment metrics
        metrics = get_experiment_service().get_experiment_metrics(experiment_id)
        
        # Group metrics by type and name
        metric_groups = {}
        for metric in metrics:
            key = f"{metric.metric_name}_{metric.metric_type}"
            if key not in metric_groups:
                metric_groups[key] = []
            metric_groups[key].append({
                'value': metric.value,
                'epoch': metric.epoch,
                'step': metric.step,
                'timestamp': metric.timestamp.isoformat()
            })
        
        # Get time series data for key metrics
        key_metrics = ['loss', 'val_loss', 'perplexity', 'bleu_score', 'rouge_1']
        metric_timeseries = {}
        
        for metric_name in key_metrics:
            timeseries = get_experiment_service().get_metric_timeseries(
                experiment_id, metric_name, 'training'
            )
            if timeseries['values']:
                metric_timeseries[metric_name] = timeseries
                
            # Also get validation metrics
            val_timeseries = get_experiment_service().get_metric_timeseries(
                experiment_id, metric_name, 'validation'
            )
            if val_timeseries['values']:
                metric_timeseries[f"val_{metric_name}"] = val_timeseries
        
        # Get experiment artifacts and notes
        artifacts = experiment.artifacts
        notes = experiment.notes if hasattr(experiment, 'notes') else []
        
        # Parse JSON fields
        try:
            hyperparameters = json.loads(experiment.hyperparameters or '{}')
            runtime_settings = json.loads(experiment.runtime_settings or '{}')
            tags = json.loads(experiment.tags or '[]')
        except json.JSONDecodeError:
            hyperparameters = {}
            runtime_settings = {}
            tags = []
        
        return render_template('experiments/detail.html',
                             experiment=experiment,
                             metrics=metrics,
                             metric_groups=metric_groups,
                             metric_timeseries=metric_timeseries,
                             artifacts=artifacts,
                             notes=notes,
                             hyperparameters=hyperparameters,
                             runtime_settings=runtime_settings,
                             tags=tags)
                             
    except Exception as e:
        logger.error(f"Error loading experiment detail: {e}")
        flash(f"Error loading experiment: {str(e)}", 'error')
        return redirect(url_for('experiments.index'))

@experiment_bp.route('/<int:experiment_id>/start', methods=['POST'])
def start_experiment(experiment_id):
    """Start an experiment"""
    try:
        success = get_experiment_service().start_experiment(experiment_id)
        
        if success:
            if request.is_json:
                return jsonify({'success': True, 'message': 'Experiment started'})
            else:
                flash('Experiment started successfully!', 'success')
        else:
            error_msg = 'Failed to start experiment'
            if request.is_json:
                return jsonify({'success': False, 'error': error_msg}), 400
            else:
                flash(error_msg, 'error')
                
    except Exception as e:
        logger.error(f"Error starting experiment: {e}")
        error_msg = f"Error starting experiment: {str(e)}"
        
        if request.is_json:
            return jsonify({'success': False, 'error': error_msg}), 500
        else:
            flash(error_msg, 'error')
    
    return redirect(url_for('experiments.detail', experiment_id=experiment_id))

@experiment_bp.route('/<int:experiment_id>/stop', methods=['POST'])
def stop_experiment(experiment_id):
    """Stop an experiment"""
    try:
        success = get_experiment_service().complete_experiment(experiment_id)
        
        if success:
            if request.is_json:
                return jsonify({'success': True, 'message': 'Experiment stopped'})
            else:
                flash('Experiment stopped successfully!', 'success')
        else:
            error_msg = 'Failed to stop experiment'
            if request.is_json:
                return jsonify({'success': False, 'error': error_msg}), 400
            else:
                flash(error_msg, 'error')
                
    except Exception as e:
        logger.error(f"Error stopping experiment: {e}")
        error_msg = f"Error stopping experiment: {str(e)}"
        
        if request.is_json:
            return jsonify({'success': False, 'error': error_msg}), 500
        else:
            flash(error_msg, 'error')
    
    return redirect(url_for('experiments.detail', experiment_id=experiment_id))

@experiment_bp.route('/<int:experiment_id>/favorite', methods=['POST'])
def toggle_favorite(experiment_id):
    """Toggle experiment favorite status"""
    try:
        is_favorite = get_experiment_service().toggle_favorite(experiment_id)
        
        if request.is_json:
            return jsonify({
                'success': True,
                'is_favorite': is_favorite,
                'message': 'Favorite status updated'
            })
        else:
            flash('Favorite status updated!', 'success')
            return redirect(url_for('experiments.detail', experiment_id=experiment_id))
            
    except Exception as e:
        logger.error(f"Error toggling favorite: {e}")
        error_msg = f"Error updating favorite: {str(e)}"
        
        if request.is_json:
            return jsonify({'success': False, 'error': error_msg}), 500
        else:
            flash(error_msg, 'error')
            return redirect(url_for('experiments.detail', experiment_id=experiment_id))

@experiment_bp.route('/<int:experiment_id>/archive', methods=['POST'])
def archive_experiment(experiment_id):
    """Archive an experiment"""
    try:
        success = get_experiment_service().archive_experiment(experiment_id)
        
        if success:
            if request.is_json:
                return jsonify({'success': True, 'message': 'Experiment archived'})
            else:
                flash('Experiment archived successfully!', 'success')
                return redirect(url_for('experiments.index'))
        else:
            error_msg = 'Failed to archive experiment'
            if request.is_json:
                return jsonify({'success': False, 'error': error_msg}), 400
            else:
                flash(error_msg, 'error')
                return redirect(url_for('experiments.detail', experiment_id=experiment_id))
                
    except Exception as e:
        logger.error(f"Error archiving experiment: {e}")
        error_msg = f"Error archiving experiment: {str(e)}"
        
        if request.is_json:
            return jsonify({'success': False, 'error': error_msg}), 500
        else:
            flash(error_msg, 'error')
            return redirect(url_for('experiments.detail', experiment_id=experiment_id))

# Comparison Routes
@experiment_bp.route('/compare')
def compare():
    """Experiment comparison interface"""
    try:
        # Get experiment IDs to compare from query params
        experiment_ids = request.args.getlist('experiments')
        experiment_ids = [int(id) for id in experiment_ids if id.isdigit()]
        
        if len(experiment_ids) < 2:
            flash('Please select at least 2 experiments to compare', 'warning')
            return redirect(url_for('experiments.index'))
        
        # Get experiments
        experiments = Experiment.query.filter(Experiment.id.in_(experiment_ids)).all()
        
        if len(experiments) != len(experiment_ids):
            flash('Some selected experiments were not found', 'error')
            return redirect(url_for('experiments.index'))
        
        # Get comparison metrics
        metrics_to_compare = ['loss', 'val_loss', 'perplexity', 'bleu_score', 'rouge_1']
        
        # Generate comparison analysis
        comparison = get_experiment_service().create_comparison(
            experiment_ids=experiment_ids,
            comparison_name=f"Comparison of {len(experiment_ids)} experiments",
            metrics_to_compare=metrics_to_compare,
            created_by=session.get('user_id'),
            description=f"Automatic comparison generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        
        # Get chart data for visualization
        chart_data = get_experiment_service().get_comparison_chart_data(comparison.id)
        
        return render_template('experiments/compare.html',
                             experiments=experiments,
                             comparison=comparison,
                             chart_data=chart_data,
                             metrics_to_compare=metrics_to_compare)
                             
    except Exception as e:
        logger.error(f"Error creating comparison: {e}")
        flash(f"Error creating comparison: {str(e)}", 'error')
        return redirect(url_for('experiments.index'))

@experiment_bp.route('/api/metrics/<int:experiment_id>')
def api_get_metrics(experiment_id):
    """API endpoint to get experiment metrics as JSON"""
    try:
        metric_name = request.args.get('metric', 'loss')
        metric_type = request.args.get('type', 'training')
        
        timeseries = get_experiment_service().get_metric_timeseries(
            experiment_id, metric_name, metric_type
        )
        
        return jsonify({
            'success': True,
            'data': timeseries
        })
        
    except Exception as e:
        logger.error(f"Error getting metrics API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@experiment_bp.route('/api/comparison/<int:comparison_id>/chart-data')
def api_get_comparison_data(comparison_id):
    """API endpoint to get comparison chart data"""
    try:
        chart_data = get_experiment_service().get_comparison_chart_data(comparison_id)
        
        return jsonify({
            'success': True,
            'data': chart_data
        })
        
    except Exception as e:
        logger.error(f"Error getting comparison data API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@experiment_bp.route('/<int:experiment_id>/notes', methods=['POST'])
def add_note(experiment_id):
    """Add a note to an experiment"""
    try:
        data = request.get_json() if request.is_json else request.form
        
        title = data.get('title', '')
        content = data.get('content', '')
        note_type = data.get('note_type', 'general')
        user_id = session.get('user_id')
        
        note = get_experiment_service().add_note(
            experiment_id=experiment_id,
            title=title,
            content=content,
            note_type=note_type,
            created_by=user_id
        )
        
        if request.is_json:
            return jsonify({
                'success': True,
                'note_id': note.id,
                'message': 'Note added successfully'
            })
        else:
            flash('Note added successfully!', 'success')
            return redirect(url_for('experiments.detail', experiment_id=experiment_id))
            
    except Exception as e:
        logger.error(f"Error adding note: {e}")
        error_msg = f"Error adding note: {str(e)}"
        
        if request.is_json:
            return jsonify({'success': False, 'error': error_msg}), 500
        else:
            flash(error_msg, 'error')
            return redirect(url_for('experiments.detail', experiment_id=experiment_id))

# Note: Sample experiments are initialized in app.py