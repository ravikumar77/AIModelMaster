"""
Prompt Playground API Routes - RESTful endpoints for prompt management and testing
"""
import json
import logging
from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash
from app import db
from models import PromptTemplate, PromptSession, PromptGeneration, PromptExport, LLMModel
from prompt_playground_service import prompt_playground_service
from auth_service import require_api_key

logger = logging.getLogger(__name__)

def init_prompt_playground_routes(app):
    """Initialize Prompt Playground routes"""
    
    # Web Interface Routes
    @app.route('/playground')
    def playground_index():
        """Prompt Playground main page"""
        sessions = prompt_playground_service.get_sessions()
        templates = prompt_playground_service.get_templates(include_public=True)
        models = LLMModel.query.filter_by(status='AVAILABLE').all()
        
        return render_template('playground/index.html', 
                             sessions=sessions, 
                             templates=templates,
                             models=models)
    
    @app.route('/playground/session/<int:session_id>')
    def playground_session(session_id):
        """Prompt session detail page"""
        session = PromptSession.query.get_or_404(session_id)
        generations = PromptGeneration.query.filter_by(session_id=session_id)\
                                          .order_by(PromptGeneration.created_at.desc()).all()
        analytics = prompt_playground_service.get_session_analytics(session_id)
        templates = prompt_playground_service.get_templates(include_public=True)
        models = LLMModel.query.filter_by(status='AVAILABLE').all()
        
        return render_template('playground/session.html',
                             session=session,
                             generations=generations,
                             analytics=analytics,
                             templates=templates,
                             models=models)
    
    @app.route('/playground/templates')
    def playground_templates():
        """Template management page"""
        templates = prompt_playground_service.get_templates(include_public=True)
        return render_template('playground/templates.html', templates=templates)
    
    @app.route('/playground/export/<int:session_id>')
    def playground_export(session_id):
        """Export configuration page"""
        session = PromptSession.query.get_or_404(session_id)
        return render_template('playground/export.html', session=session)
    

    
    # API Routes
    
    # Template Management APIs
    @app.route('/api/playground/templates', methods=['GET'])
    @require_api_key
    def api_list_templates():
        """List prompt templates"""
        try:
            category = request.args.get('category')
            user_id = getattr(request, 'user_id', None)
            
            templates = prompt_playground_service.get_templates(
                category=category, 
                user_id=user_id, 
                include_public=True
            )
            
            return jsonify({
                'success': True,
                'templates': [{
                    'id': t.id,
                    'name': t.name,
                    'description': t.description,
                    'template_content': t.template_content,
                    'category': t.category,
                    'tags': json.loads(t.tags or '[]'),
                    'is_public': t.is_public,
                    'usage_count': t.usage_count,
                    'created_at': t.created_at.isoformat()
                } for t in templates]
            })
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/playground/templates', methods=['POST'])
    @require_api_key
    def api_create_template():
        """Create new prompt template"""
        try:
            data = request.json
            user_id = getattr(request, 'user_id', None)
            
            template = prompt_playground_service.create_template(
                name=data['name'],
                template_content=data['template_content'],
                description=data.get('description', ''),
                category=data.get('category', 'general'),
                tags=data.get('tags', []),
                is_public=data.get('is_public', False),
                created_by=user_id
            )
            
            return jsonify({
                'success': True,
                'template': {
                    'id': template.id,
                    'name': template.name,
                    'description': template.description,
                    'template_content': template.template_content,
                    'category': template.category,
                    'created_at': template.created_at.isoformat()
                }
            })
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # Session Management APIs
    @app.route('/api/playground/sessions', methods=['GET'])
    @require_api_key
    def api_list_sessions():
        """List prompt sessions"""
        try:
            user_id = getattr(request, 'user_id', None)
            include_favorites = request.args.get('favorites', 'false').lower() == 'true'
            
            sessions = prompt_playground_service.get_sessions(
                user_id=user_id,
                include_favorites=include_favorites
            )
            
            return jsonify({
                'success': True,
                'sessions': [{
                    'id': s.id,
                    'name': s.name,
                    'prompt_text': s.prompt_text,
                    'model_id': s.model_id,
                    'template_id': s.template_id,
                    'parameters': {
                        'temperature': s.temperature,
                        'max_length': s.max_length,
                        'top_p': s.top_p,
                        'top_k': s.top_k,
                        'repetition_penalty': s.repetition_penalty
                    },
                    'is_favorite': s.is_favorite,
                    'tags': json.loads(s.tags or '[]'),
                    'created_at': s.created_at.isoformat(),
                    'updated_at': s.updated_at.isoformat()
                } for s in sessions]
            })
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/playground/sessions', methods=['POST'])
    @require_api_key
    def api_create_session():
        """Create new prompt session"""
        try:
            data = request.json
            user_id = getattr(request, 'user_id', None)
            
            session = prompt_playground_service.create_session(
                name=data['name'],
                prompt_text=data['prompt_text'],
                model_id=data['model_id'],
                template_id=data.get('template_id'),
                created_by=user_id,
                parameters=data.get('parameters', {})
            )
            
            return jsonify({
                'success': True,
                'session': {
                    'id': session.id,
                    'name': session.name,
                    'prompt_text': session.prompt_text,
                    'model_id': session.model_id,
                    'created_at': session.created_at.isoformat()
                }
            })
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/playground/sessions/<int:session_id>', methods=['PUT'])
    @require_api_key
    def api_update_session(session_id):
        """Update prompt session"""
        try:
            data = request.json
            session = prompt_playground_service.update_session(session_id, **data)
            
            return jsonify({
                'success': True,
                'session': {
                    'id': session.id,
                    'name': session.name,
                    'prompt_text': session.prompt_text,
                    'updated_at': session.updated_at.isoformat()
                }
            })
        except Exception as e:
            logger.error(f"Error updating session: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/playground/sessions/<int:session_id>/favorite', methods=['POST'])
    @require_api_key
    def api_toggle_favorite(session_id):
        """Toggle session favorite status"""
        try:
            is_favorite = prompt_playground_service.toggle_favorite(session_id)
            return jsonify({
                'success': True,
                'is_favorite': is_favorite
            })
        except Exception as e:
            logger.error(f"Error toggling favorite: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # Text Generation APIs
    @app.route('/api/playground/sessions/<int:session_id>/generate', methods=['POST'])
    @require_api_key
    def api_generate_text(session_id):
        """Generate text using session configuration"""
        try:
            data = request.json or {}
            input_text = data.get('input_text')
            override_params = data.get('parameters')
            
            generation = prompt_playground_service.generate_text(
                session_id=session_id,
                input_text=input_text,
                override_params=override_params
            )
            
            return jsonify({
                'success': True,
                'generation': {
                    'id': generation.id,
                    'input_text': generation.input_text,
                    'generated_text': generation.generated_text,
                    'full_prompt': generation.full_prompt,
                    'generation_time': generation.generation_time,
                    'tokens_generated': generation.tokens_generated,
                    'tokens_per_second': generation.tokens_per_second,
                    'created_at': generation.created_at.isoformat()
                }
            })
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/playground/generations/<int:generation_id>/rate', methods=['POST'])
    @require_api_key
    def api_rate_generation(generation_id):
        """Rate a generation"""
        try:
            data = request.json
            rating = data.get('rating', 0)
            flag_reason = data.get('flag_reason')
            
            success = prompt_playground_service.rate_generation(
                generation_id, rating, flag_reason
            )
            
            return jsonify({
                'success': success,
                'message': 'Rating saved successfully' if success else 'Failed to save rating'
            })
        except Exception as e:
            logger.error(f"Error rating generation: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # Export APIs with Advanced Formats
    @app.route('/api/playground/sessions/<int:session_id>/export', methods=['POST'])
    @require_api_key
    def api_export_session(session_id):
        """Export session in various formats"""
        try:
            data = request.json
            export_format = data.get('format', 'json')
            
            # Validate format
            supported_formats = ['json', 'yaml', 'curl', 'python', 'triton', 'tensorflow_lite', 'huggingface']
            if export_format not in supported_formats:
                return jsonify({
                    'success': False, 
                    'error': f'Unsupported format. Supported: {supported_formats}'
                }), 400
            
            export = prompt_playground_service.export_session(session_id, export_format)
            
            return jsonify({
                'success': True,
                'export': {
                    'id': export.id,
                    'format': export.export_format,
                    'content': export.export_content,
                    'created_at': export.created_at.isoformat()
                }
            })
        except Exception as e:
            logger.error(f"Error exporting session: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # Analytics APIs
    @app.route('/api/playground/sessions/<int:session_id>/analytics', methods=['GET'])
    @require_api_key
    def api_session_analytics(session_id):
        """Get session analytics"""
        try:
            analytics = prompt_playground_service.get_session_analytics(session_id)
            return jsonify({
                'success': True,
                'analytics': analytics
            })
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # Web Form Handlers
    @app.route('/playground/session/create', methods=['POST'])
    def create_session_form():
        """Create session from web form"""
        try:
            name = request.form.get('name')
            prompt_text = request.form.get('prompt_text')
            model_id = int(request.form.get('model_id'))
            template_id = request.form.get('template_id')
            template_id = int(template_id) if template_id else None
            
            parameters = {
                'temperature': float(request.form.get('temperature', 0.7)),
                'max_length': int(request.form.get('max_length', 100)),
                'top_p': float(request.form.get('top_p', 0.9)),
                'top_k': int(request.form.get('top_k', 50)),
                'repetition_penalty': float(request.form.get('repetition_penalty', 1.0))
            }
            
            session = prompt_playground_service.create_session(
                name=name,
                prompt_text=prompt_text,
                model_id=model_id,
                template_id=template_id,
                parameters=parameters
            )
            
            flash(f'Session "{name}" created successfully!', 'success')
            return redirect(url_for('playground_session', session_id=session.id))
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            flash(f'Error creating session: {str(e)}', 'error')
            return redirect(url_for('playground_index'))
    
    @app.route('/playground/template/create', methods=['POST'])
    def create_template_form():
        """Create template from web form"""
        try:
            name = request.form.get('name')
            description = request.form.get('description', '')
            template_content = request.form.get('template_content')
            category = request.form.get('category', 'general')
            is_public = request.form.get('is_public') == 'on'
            
            template = prompt_playground_service.create_template(
                name=name,
                description=description,
                template_content=template_content,
                category=category,
                is_public=is_public
            )
            
            flash(f'Template "{name}" created successfully!', 'success')
            return redirect(url_for('playground_templates'))
            
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            flash(f'Error creating template: {str(e)}', 'error')
            return redirect(url_for('playground_templates'))
    
    @app.route('/playground/session/<int:session_id>/generate', methods=['POST'])
    def generate_text_form(session_id):
        """Generate text from web form"""
        try:
            input_text = request.form.get('input_text', '')
            
            # Get override parameters if provided
            override_params = {}
            if request.form.get('override_temperature'):
                override_params['temperature'] = float(request.form.get('temperature', 0.7))
            if request.form.get('override_max_length'):
                override_params['max_length'] = int(request.form.get('max_length', 100))
            if request.form.get('override_top_p'):
                override_params['top_p'] = float(request.form.get('top_p', 0.9))
            if request.form.get('override_top_k'):
                override_params['top_k'] = int(request.form.get('top_k', 50))
            
            generation = prompt_playground_service.generate_text(
                session_id=session_id,
                input_text=input_text,
                override_params=override_params if override_params else None
            )
            
            flash('Text generated successfully!', 'success')
            return redirect(url_for('playground_session', session_id=session_id))
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            flash(f'Error generating text: {str(e)}', 'error')
            return redirect(url_for('playground_session', session_id=session_id))