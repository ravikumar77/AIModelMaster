import os
import logging
from flask import Flask
from db import db
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def create_app():
    # Create the app
    app = Flask(__name__)
    app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

    # Use fixed path for database
    fixed_instance_dir = r"D:\Work\instance"
    try:
        os.makedirs(fixed_instance_dir, exist_ok=True)
        logging.info(f"Fixed instance directory ensured at: {fixed_instance_dir}")
        if os.access(fixed_instance_dir, os.W_OK):
            logging.info(f"Fixed instance directory is writable: {fixed_instance_dir}")
        else:
            logging.error(f"Fixed instance directory is NOT writable: {fixed_instance_dir}")
    except Exception as e:
        logging.error(f"Failed to create fixed instance directory: {e}")
        raise

    # Configure the database
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        db_path = os.path.abspath(os.path.join(fixed_instance_dir, 'llm_platform.db'))
        logging.info(f"Resolved SQLite DB path: {db_path}")
        if os.path.exists(db_path):
            if os.access(db_path, os.W_OK):
                logging.info(f"Database file is writable: {db_path}")
            else:
                logging.error(f"Database file is NOT writable: {db_path}")
        else:
            try:
                with open(db_path, 'a') as f:
                    pass
                logging.info(f"Database file can be created: {db_path}")
            except Exception as e:
                logging.error(f"Cannot create database file at {db_path}: {e}")
        database_url = f"sqlite:///{db_path}"
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }

    # Initialize the app with the extension
    db.init_app(app)

    with app.app_context():
        # Import routes and models here to avoid circular imports
        import models
        from routes import init_routes
        # Initialize routes
        init_routes(app)
        # Create all tables
        try:
            db.create_all()
        except Exception as e:
            logging.warning(f"Could not create tables: {e}")

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
