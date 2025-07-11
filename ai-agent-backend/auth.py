from flask import request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import jwt
import datetime
import os
import json

class AuthManager:
    def __init__(self, app, db):
        self.app = app
        self.db = db
        self.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')
        
    def generate_token(self, user_id):
        """Generate JWT token for user"""
        payload = {
            'user_id': user_id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token):
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload['user_id']
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def require_auth(self, f):
        """Decorator to require authentication"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not token:
                return jsonify({'error': 'No token provided'}), 401
            
            if token.startswith('Bearer '):
                token = token[7:]
            
            user_id = self.verify_token(token)
            if not user_id:
                return jsonify({'error': 'Invalid or expired token'}), 401
            
            request.current_user_id = user_id
            return f(*args, **kwargs)
        return decorated_function
    
    def register_routes(self):
        """Register authentication routes"""
        
        @self.app.route('/api/auth/register', methods=['POST'])
        def register():
            from models import User, UserPreference
            
            data = request.get_json()
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
            
            if not all([username, email, password]):
                return jsonify({'error': 'Missing required fields'}), 400
            
            # Check if user exists
            if User.query.filter_by(email=email).first():
                return jsonify({'error': 'Email already registered'}), 400
            
            if User.query.filter_by(username=username).first():
                return jsonify({'error': 'Username already taken'}), 400
            
            # Create new user
            hashed_password = generate_password_hash(password)
            user = User(username=username, email=email, password_hash=hashed_password)
            self.db.session.add(user)
            self.db.session.commit()
            
            # Create default preferences
            preferences = UserPreference(user_id=user.id)
            self.db.session.add(preferences)
            self.db.session.commit()
            
            token = self.generate_token(user.id)
            
            return jsonify({
                'message': 'User registered successfully',
                'token': token,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email
                }
            }), 201
        
        @self.app.route('/api/auth/login', methods=['POST'])
        def login():
            from models import User
            
            data = request.get_json()
            email = data.get('email')
            password = data.get('password')
            
            if not all([email, password]):
                return jsonify({'error': 'Email and password required'}), 400
            
            user = User.query.filter_by(email=email).first()
            
            if not user or not check_password_hash(user.password_hash, password):
                return jsonify({'error': 'Invalid credentials'}), 401
            
            token = self.generate_token(user.id)
            
            return jsonify({
                'message': 'Login successful',
                'token': token,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email
                }
            })
        
        @self.app.route('/api/auth/profile', methods=['GET'])
        @self.require_auth
        def get_profile():
            from models import User, UserPreference
            
            user = User.query.get(request.current_user_id)
            preferences = UserPreference.query.filter_by(user_id=user.id).first()
            
            return jsonify({
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'created_at': user.created_at.isoformat()
                },
                'preferences': {
                    'min_price': preferences.min_price if preferences else 0,
                    'max_price': preferences.max_price if preferences else 10000,
                    'currency': preferences.currency if preferences else 'USD',
                    'language': preferences.language if preferences else 'en'
                }
            })
        
        @self.app.route('/api/auth/preferences', methods=['PUT'])
        @self.require_auth
        def update_preferences():
            from models import UserPreference
            
            data = request.get_json()
            preferences = UserPreference.query.filter_by(user_id=request.current_user_id).first()
            
            if not preferences:
                preferences = UserPreference(user_id=request.current_user_id)
                self.db.session.add(preferences)
            
            # Update preferences
            if 'min_price' in data:
                preferences.min_price = data['min_price']
            if 'max_price' in data:
                preferences.max_price = data['max_price']
            if 'currency' in data:
                preferences.currency = data['currency']
            if 'language' in data:
                preferences.language = data['language']
            if 'preferred_brands' in data:
                preferences.preferred_brands = json.dumps(data['preferred_brands'])
            
            self.db.session.commit()
            
            return jsonify({'message': 'Preferences updated successfully'})
