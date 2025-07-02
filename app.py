import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1) # needed for url_for to generate with https

# configure the database, relative to the app instance folder
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
# initialize the app with the extension, flask-sqlalchemy >= 3.0.x
db.init_app(app)

with app.app_context():
    # Make sure to import the models here or their tables won't be created
    import models  # noqa: F401
    from routes import init_routes
    from prompt_playground_routes import init_prompt_playground_routes
    from prompt_playground_service import prompt_playground_service
    from experiment_routes import experiment_bp
    from huggingface_routes import init_huggingface_routes
    
    # Initialize routes and blueprints
    init_routes(app)
    
    app.register_blueprint(experiment_bp)
    init_prompt_playground_routes(app)
    init_huggingface_routes(app)
    
    from export_routes import init_export_routes
    init_export_routes(app)
    
    from custom_routes import init_custom_routes
    init_custom_routes(app)

    db.create_all()
    
    # Initialize default templates
    try:
        prompt_playground_service.initialize_default_templates()
    except Exception as e:
        print(f"Warning: Could not initialize default templates: {e}")
    
    # Initialize sample experiments
    try:
        from experiment_tracking_service import ExperimentTrackingService
        exp_service = ExperimentTrackingService()
        exp_service.initialize_sample_experiments()
    except Exception as e:
        print(f"Warning: Could not initialize sample experiments: {e}")
