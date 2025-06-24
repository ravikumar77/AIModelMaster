"""
Authentication and API Key management service
"""
import secrets
import hashlib
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, g
from werkzeug.security import generate_password_hash, check_password_hash
from app import db
from models import User, APIKey, GenerationLog

class AuthService:
    def __init__(self):
        pass
    
    def generate_api_key(self):
        """Generate a secure API key"""
        return f"llm_{secrets.token_urlsafe(32)}"
    
    def hash_password(self, password):
        """Hash a password"""
        return generate_password_hash(password)
    
    def verify_password(self, password_hash, password):
        """Verify a password against its hash"""
        return check_password_hash(password_hash, password)
    
    def create_user(self, username, email, password):
        """Create a new user"""
        try:
            # Check if user already exists
            existing_user = User.query.filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                return None, "User already exists"
            
            # Create new user
            user = User(
                username=username,
                email=email,
                password_hash=self.hash_password(password)
            )
            
            db.session.add(user)
            db.session.commit()
            
            # Create default API key
            api_key = self.create_api_key(user.id, "Default Key")
            
            return user, api_key
            
        except Exception as e:
            db.session.rollback()
            return None, str(e)
    
    def create_api_key(self, user_id, key_name, rate_limit=1000):
        """Create a new API key for a user"""
        try:
            api_key = APIKey(
                user_id=user_id,
                key_name=key_name,
                key_value=self.generate_api_key(),
                rate_limit=rate_limit
            )
            
            db.session.add(api_key)
            db.session.commit()
            
            return api_key
            
        except Exception as e:
            db.session.rollback()
            return None
    
    def validate_api_key(self, key_value):
        """Validate an API key and return associated user"""
        try:
            api_key = APIKey.query.filter_by(
                key_value=key_value,
                is_active=True
            ).first()
            
            if not api_key:
                return None, "Invalid API key"
            
            # Check rate limiting (simple daily limit)
            today = datetime.utcnow().date()
            daily_usage = GenerationLog.query.filter(
                GenerationLog.api_key_id == api_key.id,
                db.func.date(GenerationLog.created_at) == today
            ).count()
            
            if daily_usage >= api_key.rate_limit:
                return None, "Rate limit exceeded"
            
            # Update usage stats
            api_key.last_used = datetime.utcnow()
            api_key.usage_count += 1
            db.session.commit()
            
            return api_key, None
            
        except Exception as e:
            return None, str(e)
    
    def deactivate_api_key(self, key_id, user_id):
        """Deactivate an API key"""
        try:
            api_key = APIKey.query.filter_by(
                id=key_id,
                user_id=user_id
            ).first()
            
            if api_key:
                api_key.is_active = False
                db.session.commit()
                return True
            
            return False
            
        except Exception as e:
            db.session.rollback()
            return False

# Authentication decorator for API endpoints
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for API key in headers
        api_key = request.headers.get('X-API-Key') or request.headers.get('Authorization')
        
        if api_key and api_key.startswith('Bearer '):
            api_key = api_key[7:]  # Remove 'Bearer ' prefix
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        auth_service = AuthService()
        key_obj, error = auth_service.validate_api_key(api_key)
        
        if error:
            return jsonify({'error': error}), 401
        
        # Store API key info in request context
        g.api_key = key_obj
        g.user_id = key_obj.user_id
        
        return f(*args, **kwargs)
    
    return decorated_function

# Initialize auth service
auth_service = AuthService()