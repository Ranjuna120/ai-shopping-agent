"""
Machine Learning Recommendations Engine for AI Shopping Agent
Provides personalized product recommendations using various ML algorithms
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sqlite3
from datetime import datetime, timedelta
import json
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import re
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RecommendationResult:
    """Data class for recommendation results"""
    product_name: str
    predicted_score: float
    confidence: float
    reasons: List[str]
    category: str
    estimated_price: float

class FeatureExtractor:
    """
    Extract and engineer features for machine learning models
    """
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def extract_text_features(self, product_names: List[str]) -> np.ndarray:
        """Extract TF-IDF features from product names"""
        if not self.is_fitted:
            return self.tfidf_vectorizer.fit_transform(product_names).toarray()
        else:
            return self.tfidf_vectorizer.transform(product_names).toarray()
    
    def extract_price_features(self, prices: List[float]) -> np.ndarray:
        """Extract price-based features"""
        prices_array = np.array(prices).reshape(-1, 1)
        
        # Create price bins and additional features
        price_features = []
        for price in prices:
            features = [
                price,
                np.log1p(price),  # Log price
                1 if price < 50 else 0,  # Budget category
                1 if 50 <= price < 200 else 0,  # Mid-range
                1 if 200 <= price < 500 else 0,  # Premium
                1 if price >= 500 else 0,  # Luxury
            ]
            price_features.append(features)
        
        return np.array(price_features)
    
    def extract_temporal_features(self, timestamps: List[datetime]) -> np.ndarray:
        """Extract temporal features from timestamps"""
        temporal_features = []
        
        for ts in timestamps:
            features = [
                ts.hour,  # Hour of day
                ts.weekday(),  # Day of week
                ts.month,  # Month
                1 if ts.weekday() < 5 else 0,  # Weekday
                1 if ts.hour < 12 else 0,  # Morning
                1 if 12 <= ts.hour < 18 else 0,  # Afternoon
                1 if ts.hour >= 18 else 0,  # Evening
            ]
            temporal_features.append(features)
        
        return np.array(temporal_features)
    
    def extract_category_features(self, categories: List[str]) -> np.ndarray:
        """Extract category features using label encoding"""
        if 'category' not in self.label_encoders:
            self.label_encoders['category'] = LabelEncoder()
            encoded = self.label_encoders['category'].fit_transform(categories)
        else:
            encoded = self.label_encoders['category'].transform(categories)
        
        # One-hot encode categories
        unique_categories = len(self.label_encoders['category'].classes_)
        one_hot = np.zeros((len(categories), unique_categories))
        for i, cat_idx in enumerate(encoded):
            one_hot[i, cat_idx] = 1
        
        return one_hot
    
    def extract_platform_features(self, platforms: List[str]) -> np.ndarray:
        """Extract platform features"""
        if 'platform' not in self.label_encoders:
            self.label_encoders['platform'] = LabelEncoder()
            encoded = self.label_encoders['platform'].fit_transform(platforms)
        else:
            encoded = self.label_encoders['platform'].transform(platforms)
        
        # One-hot encode platforms
        unique_platforms = len(self.label_encoders['platform'].classes_)
        one_hot = np.zeros((len(platforms), unique_platforms))
        for i, plat_idx in enumerate(encoded):
            one_hot[i, plat_idx] = 1
        
        return one_hot
    
    def combine_features(self, product_names: List[str], prices: List[float], 
                        timestamps: List[datetime], categories: List[str], 
                        platforms: List[str]) -> np.ndarray:
        """Combine all features into a single feature matrix"""
        
        text_features = self.extract_text_features(product_names)
        price_features = self.extract_price_features(prices)
        temporal_features = self.extract_temporal_features(timestamps)
        category_features = self.extract_category_features(categories)
        platform_features = self.extract_platform_features(platforms)
        
        # Combine all features
        combined_features = np.hstack([
            text_features,
            price_features,
            temporal_features,
            category_features,
            platform_features
        ])
        
        if not self.is_fitted:
            combined_features = self.scaler.fit_transform(combined_features)
            self.is_fitted = True
        else:
            combined_features = self.scaler.transform(combined_features)
        
        return combined_features

class UserProfiler:
    """
    Build comprehensive user profiles for personalized recommendations
    """
    
    def __init__(self, db_path: str = 'shopping_agent.db'):
        self.db_path = db_path
        
    def get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def build_user_profile(self, user_id: int) -> Dict[str, Any]:
        """Build comprehensive user profile"""
        conn = self.get_db_connection()
        
        # Get user data
        user_data = self._get_user_data(conn, user_id)
        search_history = self._get_search_history(conn, user_id)
        favorites = self._get_favorites(conn, user_id)
        price_alerts = self._get_price_alerts(conn, user_id)
        
        conn.close()
        
        # Build profile
        profile = {
            'user_id': user_id,
            'preferences': self._extract_preferences(search_history, favorites),
            'budget_profile': self._build_budget_profile(search_history, favorites),
            'category_preferences': self._build_category_preferences(search_history, favorites),
            'platform_preferences': self._build_platform_preferences(favorites),
            'temporal_patterns': self._build_temporal_patterns(search_history, favorites),
            'price_sensitivity': self._build_price_sensitivity_profile(price_alerts, favorites),
            'engagement_metrics': self._calculate_engagement_metrics(search_history, favorites, price_alerts)
        }
        
        return profile
    
    def _get_user_data(self, conn: sqlite3.Connection, user_id: int) -> Dict[str, Any]:
        """Get basic user data"""
        query = "SELECT * FROM users WHERE id = ?"
        result = conn.execute(query, (user_id,)).fetchone()
        
        if result:
            columns = [description[0] for description in conn.execute(query, (user_id,)).description]
            return dict(zip(columns, result))
        return {}
    
    def _get_search_history(self, conn: sqlite3.Connection, user_id: int) -> List[Dict[str, Any]]:
        """Get user search history"""
        query = """
        SELECT * FROM search_history 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 100
        """
        
        cursor = conn.execute(query, (user_id,))
        columns = [description[0] for description in cursor.description]
        
        results = []
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
        
        return results
    
    def _get_favorites(self, conn: sqlite3.Connection, user_id: int) -> List[Dict[str, Any]]:
        """Get user favorites"""
        query = """
        SELECT * FROM favorites 
        WHERE user_id = ? 
        ORDER BY added_at DESC
        """
        
        cursor = conn.execute(query, (user_id,))
        columns = [description[0] for description in cursor.description]
        
        results = []
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
        
        return results
    
    def _get_price_alerts(self, conn: sqlite3.Connection, user_id: int) -> List[Dict[str, Any]]:
        """Get user price alerts"""
        query = """
        SELECT * FROM price_alert 
        WHERE user_id = ? 
        ORDER BY created_at DESC
        """
        
        cursor = conn.execute(query, (user_id,))
        columns = [description[0] for description in cursor.description]
        
        results = []
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
        
        return results
    
    def _extract_preferences(self, search_history: List[Dict], favorites: List[Dict]) -> Dict[str, Any]:
        """Extract user preferences from behavior data"""
        all_items = []
        
        # Add search queries
        for search in search_history:
            if search.get('query'):
                all_items.append(search['query'].lower())
        
        # Add favorite product names
        for fav in favorites:
            if fav.get('product_name'):
                all_items.append(fav['product_name'].lower())
        
        # Extract keywords and patterns
        keywords = []
        for item in all_items:
            words = re.findall(r'\b\w+\b', item)
            keywords.extend(words)
        
        keyword_freq = Counter(keywords)
        
        preferences = {
            'top_keywords': dict(keyword_freq.most_common(20)),
            'preferred_terms': [word for word, count in keyword_freq.most_common(10) if count > 1],
            'search_patterns': self._identify_search_patterns(all_items)
        }
        
        return preferences
    
    def _build_budget_profile(self, search_history: List[Dict], favorites: List[Dict]) -> Dict[str, Any]:
        """Build user budget profile"""
        budgets = []
        prices = []
        
        # Extract budgets from searches
        for search in search_history:
            if search.get('budget'):
                try:
                    budget = float(search['budget'])
                    budgets.append(budget)
                except (ValueError, TypeError):
                    pass
        
        # Extract prices from favorites
        for fav in favorites:
            if fav.get('price'):
                try:
                    price = float(fav['price'])
                    prices.append(price)
                except (ValueError, TypeError):
                    pass
        
        all_values = budgets + prices
        
        if not all_values:
            return {'budget_range': 'unknown', 'spending_tier': 'unknown'}
        
        profile = {
            'average_budget': np.mean(budgets) if budgets else 0,
            'average_price': np.mean(prices) if prices else 0,
            'budget_range': {
                'min': min(all_values),
                'max': max(all_values),
                'median': np.median(all_values)
            },
            'spending_tier': self._classify_spending_tier(np.mean(all_values)),
            'budget_consistency': np.std(budgets) / np.mean(budgets) if budgets and np.mean(budgets) > 0 else 0
        }
        
        return profile
    
    def _build_category_preferences(self, search_history: List[Dict], favorites: List[Dict]) -> Dict[str, float]:
        """Build category preference scores"""
        categories = []
        
        # Extract categories from searches
        for search in search_history:
            if search.get('query'):
                category = self._categorize_product(search['query'])
                categories.append(category)
        
        # Extract categories from favorites
        for fav in favorites:
            if fav.get('product_name'):
                category = self._categorize_product(fav['product_name'])
                categories.append(category)
        
        if not categories:
            return {}
        
        category_counts = Counter(categories)
        total_items = len(categories)
        
        # Calculate preference scores (normalized)
        preferences = {}
        for category, count in category_counts.items():
            preferences[category] = count / total_items
        
        return preferences
    
    def _build_platform_preferences(self, favorites: List[Dict]) -> Dict[str, float]:
        """Build platform preference scores"""
        platforms = []
        
        for fav in favorites:
            if fav.get('platform'):
                platforms.append(fav['platform'].lower())
        
        if not platforms:
            return {}
        
        platform_counts = Counter(platforms)
        total_items = len(platforms)
        
        # Calculate preference scores
        preferences = {}
        for platform, count in platform_counts.items():
            preferences[platform] = count / total_items
        
        return preferences
    
    def _build_temporal_patterns(self, search_history: List[Dict], favorites: List[Dict]) -> Dict[str, Any]:
        """Build temporal behavior patterns"""
        timestamps = []
        
        # Extract timestamps
        for search in search_history:
            if search.get('created_at'):
                try:
                    ts = datetime.fromisoformat(search['created_at'])
                    timestamps.append(ts)
                except (ValueError, TypeError):
                    pass
        
        for fav in favorites:
            if fav.get('added_at'):
                try:
                    ts = datetime.fromisoformat(fav['added_at'])
                    timestamps.append(ts)
                except (ValueError, TypeError):
                    pass
        
        if not timestamps:
            return {}
        
        hours = [ts.hour for ts in timestamps]
        days = [ts.weekday() for ts in timestamps]
        
        patterns = {
            'peak_hours': Counter(hours).most_common(3),
            'peak_days': Counter(days).most_common(3),
            'morning_activity': len([h for h in hours if 6 <= h < 12]) / len(hours),
            'afternoon_activity': len([h for h in hours if 12 <= h < 18]) / len(hours),
            'evening_activity': len([h for h in hours if 18 <= h < 24]) / len(hours),
            'weekday_activity': len([d for d in days if d < 5]) / len(days),
            'weekend_activity': len([d for d in days if d >= 5]) / len(days)
        }
        
        return patterns
    
    def _build_price_sensitivity_profile(self, price_alerts: List[Dict], favorites: List[Dict]) -> Dict[str, Any]:
        """Build price sensitivity profile"""
        target_prices = []
        current_prices = []
        favorite_prices = []
        
        for alert in price_alerts:
            if alert.get('target_price'):
                try:
                    target_prices.append(float(alert['target_price']))
                except (ValueError, TypeError):
                    pass
            
            if alert.get('current_price'):
                try:
                    current_prices.append(float(alert['current_price']))
                except (ValueError, TypeError):
                    pass
        
        for fav in favorites:
            if fav.get('price'):
                try:
                    favorite_prices.append(float(fav['price']))
                except (ValueError, TypeError):
                    pass
        
        # Calculate sensitivity metrics
        profile = {
            'alert_frequency': len(price_alerts) / 30,  # alerts per day
            'average_discount_sought': 0,
            'sensitivity_score': 0.5,  # default neutral
            'price_consciousness': 'medium'
        }
        
        if target_prices and current_prices:
            discounts = [(cp - tp) / cp for cp, tp in zip(current_prices, target_prices) if cp > 0]
            if discounts:
                profile['average_discount_sought'] = np.mean(discounts)
                profile['sensitivity_score'] = min(np.mean(discounts) * 2, 1.0)
        
        # Classify price consciousness
        if profile['sensitivity_score'] > 0.7:
            profile['price_consciousness'] = 'high'
        elif profile['sensitivity_score'] < 0.3:
            profile['price_consciousness'] = 'low'
        
        return profile
    
    def _calculate_engagement_metrics(self, search_history: List[Dict], favorites: List[Dict], price_alerts: List[Dict]) -> Dict[str, float]:
        """Calculate user engagement metrics"""
        now = datetime.now()
        month_ago = now - timedelta(days=30)
        
        # Count recent activities
        recent_searches = len([s for s in search_history if s.get('created_at') and datetime.fromisoformat(s['created_at']) > month_ago])
        recent_favorites = len([f for f in favorites if f.get('added_at') and datetime.fromisoformat(f['added_at']) > month_ago])
        recent_alerts = len([a for a in price_alerts if a.get('created_at') and datetime.fromisoformat(a['created_at']) > month_ago])
        
        metrics = {
            'search_frequency': recent_searches / 30,
            'favorite_frequency': recent_favorites / 30,
            'alert_frequency': recent_alerts / 30,
            'total_engagement': (recent_searches + recent_favorites + recent_alerts) / 30,
            'engagement_level': 'low'
        }
        
        # Classify engagement level
        total_engagement = metrics['total_engagement']
        if total_engagement > 2:
            metrics['engagement_level'] = 'high'
        elif total_engagement > 0.5:
            metrics['engagement_level'] = 'medium'
        
        return metrics
    
    def _identify_search_patterns(self, items: List[str]) -> List[str]:
        """Identify common search patterns"""
        patterns = []
        
        # Look for common patterns
        brand_mentions = []
        feature_mentions = []
        
        for item in items:
            # Extract potential brand names (capitalized words)
            brands = re.findall(r'\b[A-Z][a-z]+\b', item)
            brand_mentions.extend(brands)
            
            # Extract feature keywords
            features = ['wireless', 'portable', 'smart', 'digital', 'rechargeable', 'bluetooth']
            for feature in features:
                if feature in item.lower():
                    feature_mentions.append(feature)
        
        if brand_mentions:
            patterns.append(f"Brand preference: {Counter(brand_mentions).most_common(1)[0][0]}")
        
        if feature_mentions:
            patterns.append(f"Feature interest: {Counter(feature_mentions).most_common(1)[0][0]}")
        
        return patterns
    
    def _classify_spending_tier(self, average_amount: float) -> str:
        """Classify user spending tier"""
        if average_amount < 50:
            return 'budget'
        elif average_amount < 200:
            return 'moderate'
        elif average_amount < 500:
            return 'premium'
        else:
            return 'luxury'
    
    def _categorize_product(self, product_name: str) -> str:
        """Categorize product based on name"""
        if not product_name:
            return 'unknown'
        
        product_lower = product_name.lower()
        
        categories = {
            'electronics': ['laptop', 'computer', 'phone', 'smartphone', 'tablet', 'camera', 'headphone', 'speaker', 'mouse', 'keyboard'],
            'clothing': ['shirt', 'pant', 'dress', 'shoe', 'jacket', 'coat', 'sweater', 'jeans', 'top'],
            'home': ['furniture', 'chair', 'table', 'bed', 'sofa', 'lamp', 'rug', 'kitchen', 'bathroom'],
            'sports': ['fitness', 'exercise', 'sport', 'gym', 'outdoor', 'bike', 'yoga', 'running'],
            'books': ['book', 'novel', 'textbook', 'kindle', 'magazine'],
            'beauty': ['makeup', 'skincare', 'beauty', 'cosmetic', 'perfume', 'lotion'],
            'automotive': ['car', 'auto', 'tire', 'battery', 'oil'],
            'gaming': ['game', 'gaming', 'console', 'controller', 'playstation', 'xbox'],
            'toys': ['toy', 'doll', 'puzzle', 'lego', 'children'],
            'jewelry': ['jewelry', 'ring', 'necklace', 'watch', 'bracelet']
        }
        
        for category, keywords in categories.items():
            if any(keyword in product_lower for keyword in keywords):
                return category
        
        return 'other'

class RecommendationEngine:
    """
    Main recommendation engine using multiple ML approaches
    """
    
    def __init__(self, db_path: str = 'shopping_agent.db'):
        self.db_path = db_path
        self.feature_extractor = FeatureExtractor()
        self.user_profiler = UserProfiler(db_path)
        self.models = {}
        self.model_paths = {
            'collaborative_filter': 'models/collaborative_filter.pkl',
            'content_based': 'models/content_based.pkl',
            'hybrid': 'models/hybrid_model.pkl'
        }
        self._ensure_model_directory()
        
    def _ensure_model_directory(self):
        """Ensure model directory exists"""
        os.makedirs('models', exist_ok=True)
    
    def get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def train_models(self):
        """Train all recommendation models"""
        logger.info("Starting model training...")
        
        # Get training data
        training_data = self._prepare_training_data()
        
        if not training_data['features'].size:
            logger.warning("No training data available")
            return
        
        # Train content-based model
        self._train_content_based_model(training_data)
        
        # Train collaborative filtering model
        self._train_collaborative_model(training_data)
        
        # Train hybrid model
        self._train_hybrid_model(training_data)
        
        logger.info("Model training completed")
    
    def _prepare_training_data(self) -> Dict[str, Any]:
        """Prepare training data from database"""
        conn = self.get_db_connection()
        
        try:
            # Get all products from favorites and search history
            query = """
            SELECT f.product_name, f.price, f.platform, f.user_id, f.added_at,
                   'favorite' as interaction_type, 1.0 as rating
            FROM favorites f
            UNION ALL
            SELECT sh.query as product_name, sh.budget as price, 'search' as platform, 
                   sh.user_id, sh.created_at as added_at,
                   'search' as interaction_type, 0.5 as rating
            FROM search_history sh
            WHERE sh.query IS NOT NULL
            """
            
            df = pd.read_sql_query(query, conn)
        except Exception as e:
            logger.warning(f"Could not load training data from database: {e}")
            df = pd.DataFrame()  # Return empty DataFrame if tables don't exist
        finally:
            conn.close()
        
        if df.empty:
            return {'features': np.array([]), 'labels': np.array([]), 'user_ids': np.array([])}
        
        # Prepare features
        product_names = df['product_name'].fillna('').tolist()
        prices = df['price'].fillna(0).astype(float).tolist()
        timestamps = pd.to_datetime(df['added_at']).tolist()
        platforms = df['platform'].fillna('unknown').tolist()
        
        # Categorize products
        categories = [self.user_profiler._categorize_product(name) for name in product_names]
        
        # Extract features
        features = self.feature_extractor.combine_features(
            product_names, prices, timestamps, categories, platforms
        )
        
        # Prepare labels (ratings)
        labels = df['rating'].values
        user_ids = df['user_id'].values
        
        return {
            'features': features,
            'labels': labels,
            'user_ids': user_ids,
            'product_names': product_names,
            'categories': categories,
            'prices': prices
        }
    
    def _train_content_based_model(self, training_data: Dict[str, Any]):
        """Train content-based recommendation model"""
        if training_data['features'].size == 0:
            return
        
        # Use Random Forest for content-based recommendations
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        X = training_data['features']
        y = training_data['labels']
        
        if len(X) > 1:
            model.fit(X, y)
            
            # Save model
            with open(self.model_paths['content_based'], 'wb') as f:
                pickle.dump(model, f)
            
            self.models['content_based'] = model
            logger.info("Content-based model trained successfully")
    
    def _train_collaborative_model(self, training_data: Dict[str, Any]):
        """Train collaborative filtering model"""
        if training_data['features'].size == 0:
            return
        
        # Create user-item matrix
        user_ids = training_data['user_ids']
        product_names = training_data['product_names']
        ratings = training_data['labels']
        
        # Simple matrix factorization approach
        unique_users = list(set(user_ids))
        unique_products = list(set(product_names))
        
        user_product_matrix = np.zeros((len(unique_users), len(unique_products)))
        
        for user_id, product, rating in zip(user_ids, product_names, ratings):
            user_idx = unique_users.index(user_id)
            product_idx = unique_products.index(product)
            user_product_matrix[user_idx, product_idx] = rating
        
        # Use KMeans for user clustering (simple collaborative approach)
        if len(unique_users) > 1:
            n_clusters = min(5, len(unique_users))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            user_clusters = kmeans.fit_predict(user_product_matrix)
            
            # Save model
            model_data = {
                'kmeans': kmeans,
                'user_product_matrix': user_product_matrix,
                'unique_users': unique_users,
                'unique_products': unique_products,
                'user_clusters': user_clusters
            }
            
            with open(self.model_paths['collaborative_filter'], 'wb') as f:
                pickle.dump(model_data, f)
            
            self.models['collaborative_filter'] = model_data
            logger.info("Collaborative filtering model trained successfully")
    
    def _train_hybrid_model(self, training_data: Dict[str, Any]):
        """Train hybrid recommendation model"""
        if training_data['features'].size == 0:
            return
        
        # Combine features with user information
        X = training_data['features']
        y = training_data['labels']
        
        if len(X) > 1:
            # Use ensemble approach
            models = {
                'rf': RandomForestRegressor(n_estimators=50, random_state=42),
                'lr': LinearRegression()
            }
            
            trained_models = {}
            for name, model in models.items():
                try:
                    model.fit(X, y)
                    trained_models[name] = model
                except Exception as e:
                    logger.warning(f"Failed to train {name}: {e}")
            
            if trained_models:
                # Save hybrid model
                with open(self.model_paths['hybrid'], 'wb') as f:
                    pickle.dump(trained_models, f)
                
                self.models['hybrid'] = trained_models
                logger.info("Hybrid model trained successfully")
    
    def load_models(self):
        """Load trained models from disk"""
        for model_name, path in self.model_paths.items():
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    logger.info(f"Loaded {model_name} model")
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
    
    def get_recommendations(self, user_id: int, num_recommendations: int = 10) -> List[RecommendationResult]:
        """Get personalized recommendations for a user"""
        # Build user profile
        user_profile = self.user_profiler.build_user_profile(user_id)
        
        # Get candidate products
        candidates = self._get_candidate_products(user_profile)
        
        if not candidates:
            return []
        
        # Score candidates using different approaches
        recommendations = []
        
        for candidate in candidates:
            scores = self._score_candidate(candidate, user_profile)
            
            if scores['final_score'] > 0:
                recommendation = RecommendationResult(
                    product_name=candidate['name'],
                    predicted_score=scores['final_score'],
                    confidence=scores['confidence'],
                    reasons=scores['reasons'],
                    category=candidate['category'],
                    estimated_price=candidate['estimated_price']
                )
                recommendations.append(recommendation)
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x.predicted_score, reverse=True)
        return recommendations[:num_recommendations]
    
    def _get_candidate_products(self, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate candidate products based on user profile"""
        candidates = []
        
        # Generate candidates based on user preferences
        category_prefs = user_profile.get('category_preferences', {})
        budget_profile = user_profile.get('budget_profile', {})
        
        # Create synthetic candidates based on popular categories and user budget
        popular_products = {
            'electronics': [
                'Wireless Bluetooth Headphones', 'Gaming Mouse', 'Laptop Stand',
                'USB-C Hub', 'Portable Speaker', 'Webcam', 'Wireless Charger'
            ],
            'clothing': [
                'Cotton T-Shirt', 'Jeans', 'Running Shoes', 'Winter Jacket',
                'Dress Shirt', 'Sneakers', 'Hoodie'
            ],
            'home': [
                'LED Desk Lamp', 'Coffee Maker', 'Storage Box',
                'Throw Pillow', 'Picture Frame', 'Kitchen Scale'
            ],
            'sports': [
                'Yoga Mat', 'Resistance Bands', 'Water Bottle',
                'Fitness Tracker', 'Dumbbells', 'Running Shorts'
            ]
        }
        
        avg_budget = budget_profile.get('average_budget', 100)
        
        for category, score in category_prefs.items():
            if category in popular_products and score > 0.1:
                for product in popular_products[category]:
                    # Estimate price based on category and user budget
                    base_prices = {
                        'electronics': avg_budget * 1.2,
                        'clothing': avg_budget * 0.8,
                        'home': avg_budget * 0.9,
                        'sports': avg_budget * 0.7
                    }
                    
                    estimated_price = base_prices.get(category, avg_budget)
                    
                    candidates.append({
                        'name': product,
                        'category': category,
                        'estimated_price': estimated_price,
                        'category_score': score
                    })
        
        return candidates
    
    def _score_candidate(self, candidate: Dict[str, Any], user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Score a candidate product for a user"""
        scores = {
            'category_score': 0,
            'price_score': 0,
            'preference_score': 0,
            'final_score': 0,
            'confidence': 0,
            'reasons': []
        }
        
        # Category preference score
        category_prefs = user_profile.get('category_preferences', {})
        category_score = category_prefs.get(candidate['category'], 0)
        scores['category_score'] = category_score
        
        if category_score > 0.2:
            scores['reasons'].append(f"Matches your interest in {candidate['category']}")
        
        # Price compatibility score
        budget_profile = user_profile.get('budget_profile', {})
        avg_budget = budget_profile.get('average_budget', 100)
        estimated_price = candidate['estimated_price']
        
        if avg_budget > 0:
            price_ratio = estimated_price / avg_budget
            if 0.5 <= price_ratio <= 1.5:  # Within reasonable range
                price_score = 1 - abs(price_ratio - 1)
                scores['price_score'] = price_score
                scores['reasons'].append(f"Price fits your budget range")
        
        # Preference matching score
        preferences = user_profile.get('preferences', {})
        preferred_terms = preferences.get('preferred_terms', [])
        
        preference_score = 0
        product_name_lower = candidate['name'].lower()
        matching_terms = [term for term in preferred_terms if term in product_name_lower]
        
        if matching_terms:
            preference_score = len(matching_terms) / len(preferred_terms) if preferred_terms else 0
            scores['preference_score'] = preference_score
            scores['reasons'].append(f"Matches your preferences: {', '.join(matching_terms[:3])}")
        
        # Calculate final score (weighted combination)
        weights = {
            'category': 0.4,
            'price': 0.3,
            'preference': 0.3
        }
        
        final_score = (
            weights['category'] * scores['category_score'] +
            weights['price'] * scores['price_score'] +
            weights['preference'] * scores['preference_score']
        )
        
        scores['final_score'] = final_score
        
        # Calculate confidence based on data availability
        confidence_factors = [
            1 if category_prefs else 0,
            1 if budget_profile.get('average_budget', 0) > 0 else 0,
            1 if preferred_terms else 0
        ]
        
        scores['confidence'] = sum(confidence_factors) / len(confidence_factors)
        
        return scores

# Flask route integration
def create_ml_routes(app, db):
    """Create ML recommendation routes for Flask app"""
    engine = RecommendationEngine()
    
    # Load existing models or train new ones
    engine.load_models()
    if not engine.models:
        try:
            engine.train_models()
        except Exception as e:
            logger.warning(f"Could not train models during initialization: {e}")
            # Continue without trained models - they can be trained later
    
    @app.route('/api/ml/recommendations/<int:user_id>', methods=['GET'])
    def get_ml_recommendations(user_id):
        """Get ML-based recommendations for user"""
        try:
            from flask import request
            num_recs = request.args.get('limit', 10, type=int)
            
            recommendations = engine.get_recommendations(user_id, num_recs)
            
            # Convert to serializable format
            recs_data = []
            for rec in recommendations:
                recs_data.append({
                    'product_name': rec.product_name,
                    'predicted_score': rec.predicted_score,
                    'confidence': rec.confidence,
                    'reasons': rec.reasons,
                    'category': rec.category,
                    'estimated_price': rec.estimated_price
                })
            
            return {
                'success': True,
                'recommendations': recs_data,
                'total_count': len(recs_data)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500
    
    @app.route('/api/ml/retrain', methods=['POST'])
    def retrain_models():
        """Retrain ML models with latest data"""
        try:
            engine.train_models()
            engine.load_models()
            
            return {
                'success': True,
                'message': 'Models retrained successfully',
                'available_models': list(engine.models.keys())
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500
    
    @app.route('/api/ml/user-profile/<int:user_id>', methods=['GET'])
    def get_user_profile(user_id):
        """Get user profile for recommendations"""
        try:
            profile = engine.user_profiler.build_user_profile(user_id)
            return {
                'success': True,
                'profile': profile
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500

if __name__ == "__main__":
    # Test the ML engine
    engine = RecommendationEngine()
    engine.train_models()
    recommendations = engine.get_recommendations(1, 5)
    print(f"Generated {len(recommendations)} recommendations")
    for rec in recommendations:
        print(f"- {rec.product_name} (Score: {rec.predicted_score:.2f})")
