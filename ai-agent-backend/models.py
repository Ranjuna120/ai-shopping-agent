from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    preferences = db.relationship('UserPreference', backref='user', lazy=True)
    search_history = db.relationship('SearchHistory', backref='user', lazy=True)
    favorites = db.relationship('Favorite', backref='user', lazy=True)

class UserPreference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    min_price = db.Column(db.Float, default=0)
    max_price = db.Column(db.Float, default=10000)
    preferred_brands = db.Column(db.Text)  # JSON string
    preferred_categories = db.Column(db.Text)  # JSON string
    currency = db.Column(db.String(3), default='USD')
    language = db.Column(db.String(5), default='en')

class SearchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    query = db.Column(db.String(200), nullable=False)
    budget = db.Column(db.Float)
    results_count = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    product_name = db.Column(db.String(200), nullable=False)
    product_url = db.Column(db.Text, nullable=False)
    price = db.Column(db.Float)
    platform = db.Column(db.String(50))
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

class PriceAlert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    product_url = db.Column(db.Text, nullable=False)
    target_price = db.Column(db.Float, nullable=False)
    current_price = db.Column(db.Float)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
