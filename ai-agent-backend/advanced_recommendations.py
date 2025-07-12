"""
Advanced Recommendation Algorithm System for AI Shopping Agent
Implements multiple recommendation strategies with ensemble methods
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from collections import defaultdict, Counter
import json
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
import math
import re
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RecommendationScore:
    """Individual recommendation score with details"""
    algorithm: str
    score: float
    confidence: float
    features_used: List[str]
    explanation: str

@dataclass 
class ProductRecommendation:
    """Complete product recommendation with multiple scores"""
    product_id: str
    product_name: str
    category: str
    estimated_price: float
    platform: str
    scores: List[RecommendationScore]
    final_score: float
    confidence: float
    reasons: List[str]
    similarity_items: List[str]

class ContentBasedRecommender:
    """
    Content-based recommendation algorithm using product features
    """
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, products_df: pd.DataFrame):
        """Fit the content-based model"""
        if products_df.empty:
            logger.warning("Empty products dataframe for content-based fitting")
            return
        
        # Prepare text features
        text_features = []
        for _, product in products_df.iterrows():
            text = f"{product.get('name', '')} {product.get('category', '')} {product.get('description', '')}"
            text_features.append(text.lower())
        
        if text_features:
            # Fit TF-IDF vectorizer
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
            
            # Fit LDA topic model
            self.topic_matrix = self.lda_model.fit_transform(self.tfidf_matrix)
            
            # Prepare numerical features
            numerical_features = []
            for _, product in products_df.iterrows():
                features = [
                    product.get('price', 0),
                    len(product.get('name', '')),
                    product.get('rating', 0) if 'rating' in product else 3.5,
                    product.get('review_count', 0) if 'review_count' in product else 0
                ]
                numerical_features.append(features)
            
            self.numerical_matrix = self.scaler.fit_transform(np.array(numerical_features))
            self.products_df = products_df.copy()
            self.is_fitted = True
            
            logger.info(f"Content-based model fitted with {len(products_df)} products")
    
    def get_recommendations(self, user_profile: Dict[str, Any], num_recommendations: int = 10) -> List[RecommendationScore]:
        """Get content-based recommendations"""
        if not self.is_fitted:
            return []
        
        user_preferences = user_profile.get('preferences', {})
        category_preferences = user_profile.get('category_preferences', {})
        budget_profile = user_profile.get('budget_profile', {})
        
        recommendations = []
        
        # Calculate similarity based on user preferences
        for idx, product in self.products_df.iterrows():
            score = self._calculate_content_score(product, user_preferences, category_preferences, budget_profile)
            
            if score > 0.1:  # Minimum threshold
                recommendations.append(RecommendationScore(
                    algorithm='content_based',
                    score=score,
                    confidence=min(score * 1.2, 1.0),
                    features_used=['text_similarity', 'category_match', 'price_compatibility'],
                    explanation=f"Matches your preferences in {product.get('category', 'unknown')} category"
                ))
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:num_recommendations]
    
    def _calculate_content_score(self, product: pd.Series, user_preferences: Dict, 
                                category_preferences: Dict, budget_profile: Dict) -> float:
        """Calculate content-based score for a product"""
        score = 0.0
        
        # Category preference score
        product_category = product.get('category', '').lower()
        category_score = category_preferences.get(product_category, 0)
        score += category_score * 0.4
        
        # Price compatibility score
        product_price = product.get('price', 0)
        avg_budget = budget_profile.get('average_budget', 100)
        
        if avg_budget > 0:
            price_ratio = product_price / avg_budget
            if 0.5 <= price_ratio <= 1.5:  # Within reasonable range
                price_score = 1 - abs(price_ratio - 1) * 0.5
                score += price_score * 0.3
        
        # Text similarity score (simplified)
        preferred_terms = user_preferences.get('preferred_terms', [])
        product_text = product.get('name', '').lower()
        
        matching_terms = sum(1 for term in preferred_terms if term in product_text)
        if preferred_terms:
            text_score = matching_terms / len(preferred_terms)
            score += text_score * 0.3
        
        return min(score, 1.0)

class CollaborativeFilteringRecommender:
    """
    Collaborative filtering recommendation algorithm
    """
    
    def __init__(self):
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.user_similarity_matrix = None
        self.kmeans_model = None
        self.is_fitted = False
        
    def fit(self, interactions_df: pd.DataFrame):
        """Fit the collaborative filtering model"""
        if interactions_df.empty:
            logger.warning("Empty interactions dataframe for collaborative filtering")
            return
        
        # Create user-item matrix
        self.user_item_matrix = interactions_df.pivot_table(
            index='user_id',
            columns='product_name',
            values='rating',
            fill_value=0
        )
        
        # Calculate item-item similarity
        item_features = self.user_item_matrix.T.values
        self.item_similarity_matrix = cosine_similarity(item_features)
        
        # Calculate user-user similarity
        user_features = self.user_item_matrix.values
        self.user_similarity_matrix = cosine_similarity(user_features)
        
        # User clustering for scalability
        if len(self.user_item_matrix) > 10:
            n_clusters = min(5, len(self.user_item_matrix) // 2)
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
            self.user_clusters = self.kmeans_model.fit_predict(user_features)
        
        self.is_fitted = True
        logger.info(f"Collaborative filtering model fitted with {len(self.user_item_matrix)} users")
    
    def get_recommendations(self, user_id: int, num_recommendations: int = 10) -> List[RecommendationScore]:
        """Get collaborative filtering recommendations"""
        if not self.is_fitted or user_id not in self.user_item_matrix.index:
            return []
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx].values
        
        # Find similar users
        user_similarities = self.user_similarity_matrix[user_idx]
        similar_users = np.argsort(user_similarities)[::-1][1:6]  # Top 5 similar users
        
        recommendations = []
        
        # Get items liked by similar users
        for similar_user_idx in similar_users:
            similar_user_ratings = self.user_item_matrix.iloc[similar_user_idx]
            similarity_score = user_similarities[similar_user_idx]
            
            for item_idx, rating in enumerate(similar_user_ratings.values):
                if rating > 0 and user_ratings[item_idx] == 0:  # Item not rated by user
                    item_name = self.user_item_matrix.columns[item_idx]
                    
                    score = rating * similarity_score
                    
                    recommendations.append(RecommendationScore(
                        algorithm='collaborative_filtering',
                        score=score,
                        confidence=similarity_score,
                        features_used=['user_similarity', 'item_ratings'],
                        explanation=f"Users similar to you liked this item (similarity: {similarity_score:.2f})"
                    ))
        
        # Sort and deduplicate
        recommendations.sort(key=lambda x: x.score, reverse=True)
        seen_items = set()
        unique_recommendations = []
        
        for rec in recommendations:
            if len(unique_recommendations) >= num_recommendations:
                break
            if rec.explanation not in seen_items:
                seen_items.add(rec.explanation)
                unique_recommendations.append(rec)
        
        return unique_recommendations

class MatrixFactorizationRecommender:
    """
    Matrix factorization recommendation algorithm using NMF
    """
    
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.nmf_model = NMF(n_components=n_components, random_state=42, max_iter=200)
        self.user_features = None
        self.item_features = None
        self.is_fitted = False
        
    def fit(self, interactions_df: pd.DataFrame):
        """Fit the matrix factorization model"""
        if interactions_df.empty:
            return
        
        # Create user-item matrix
        user_item_matrix = interactions_df.pivot_table(
            index='user_id',
            columns='product_name',
            values='rating',
            fill_value=0
        )
        
        # Apply NMF
        self.user_features = self.nmf_model.fit_transform(user_item_matrix.values)
        self.item_features = self.nmf_model.components_
        
        self.user_ids = user_item_matrix.index
        self.item_names = user_item_matrix.columns
        self.is_fitted = True
        
        logger.info(f"Matrix factorization model fitted with {len(user_item_matrix)} users")
    
    def get_recommendations(self, user_id: int, num_recommendations: int = 10) -> List[RecommendationScore]:
        """Get matrix factorization recommendations"""
        if not self.is_fitted or user_id not in self.user_ids:
            return []
        
        user_idx = self.user_ids.get_loc(user_id)
        user_vector = self.user_features[user_idx]
        
        # Calculate predicted ratings
        predicted_ratings = np.dot(user_vector, self.item_features)
        
        # Get top recommendations
        top_items = np.argsort(predicted_ratings)[::-1][:num_recommendations * 2]
        
        recommendations = []
        for item_idx in top_items:
            if len(recommendations) >= num_recommendations:
                break
                
            score = predicted_ratings[item_idx]
            if score > 0.1:  # Minimum threshold
                item_name = self.item_names[item_idx]
                
                recommendations.append(RecommendationScore(
                    algorithm='matrix_factorization',
                    score=score,
                    confidence=min(score * 0.8, 1.0),
                    features_used=['latent_factors'],
                    explanation=f"Predicted based on latent preference patterns (score: {score:.2f})"
                ))
        
        return recommendations

class KnowledgeBasedRecommender:
    """
    Knowledge-based recommendation using product rules and constraints
    """
    
    def __init__(self):
        self.rules = self._load_recommendation_rules()
        self.product_graph = nx.Graph()
        
    def _load_recommendation_rules(self) -> Dict[str, Any]:
        """Load recommendation rules"""
        return {
            'category_rules': {
                'electronics': {
                    'complementary': ['accessories', 'cables', 'cases'],
                    'upgrade_path': ['premium_electronics'],
                    'seasonal': {'winter': 0.8, 'summer': 1.2}
                },
                'clothing': {
                    'complementary': ['shoes', 'accessories'],
                    'seasonal': {'winter': 1.2, 'summer': 0.8, 'spring': 1.0, 'fall': 1.0}
                },
                'home': {
                    'complementary': ['decor', 'furniture'],
                    'bulk_discount': True
                }
            },
            'price_rules': {
                'luxury_threshold': 500,
                'budget_threshold': 50,
                'discount_preference': 0.2
            },
            'brand_rules': {
                'premium_brands': ['apple', 'samsung', 'sony', 'nike', 'adidas'],
                'budget_brands': ['generic', 'store_brand']
            }
        }
    
    def fit(self, products_df: pd.DataFrame):
        """Build product knowledge graph"""
        # Create product graph with relationships
        for _, product in products_df.iterrows():
            product_id = str(product.get('id', ''))
            self.product_graph.add_node(product_id, **product.to_dict())
            
        # Add edges based on category relationships
        for node1 in self.product_graph.nodes():
            for node2 in self.product_graph.nodes():
                if node1 != node2:
                    similarity = self._calculate_product_similarity(
                        self.product_graph.nodes[node1],
                        self.product_graph.nodes[node2]
                    )
                    if similarity > 0.3:
                        self.product_graph.add_edge(node1, node2, weight=similarity)
        
        logger.info(f"Knowledge graph built with {len(self.product_graph.nodes())} products")
    
    def get_recommendations(self, user_profile: Dict[str, Any], num_recommendations: int = 10) -> List[RecommendationScore]:
        """Get knowledge-based recommendations"""
        recommendations = []
        
        # Apply category rules
        category_prefs = user_profile.get('category_preferences', {})
        for category, preference_score in category_prefs.items():
            if category in self.rules['category_rules']:
                category_rec = self._apply_category_rules(category, preference_score, user_profile)
                recommendations.extend(category_rec)
        
        # Apply price rules
        budget_profile = user_profile.get('budget_profile', {})
        price_recs = self._apply_price_rules(budget_profile)
        recommendations.extend(price_recs)
        
        # Apply temporal rules
        temporal_recs = self._apply_temporal_rules(user_profile)
        recommendations.extend(temporal_recs)
        
        # Sort and return top recommendations
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:num_recommendations]
    
    def _apply_category_rules(self, category: str, preference_score: float, user_profile: Dict) -> List[RecommendationScore]:
        """Apply category-specific rules"""
        recommendations = []
        rules = self.rules['category_rules'].get(category, {})
        
        # Complementary items
        complementary_items = rules.get('complementary', [])
        for comp_category in complementary_items:
            score = preference_score * 0.7  # Reduced score for complementary
            recommendations.append(RecommendationScore(
                algorithm='knowledge_based',
                score=score,
                confidence=0.8,
                features_used=['category_rules', 'complementary_items'],
                explanation=f"Complements your interest in {category}"
            ))
        
        # Seasonal adjustments
        seasonal_factors = rules.get('seasonal', {})
        current_season = self._get_current_season()
        if current_season in seasonal_factors:
            seasonal_score = preference_score * seasonal_factors[current_season]
            recommendations.append(RecommendationScore(
                algorithm='knowledge_based',
                score=seasonal_score,
                confidence=0.9,
                features_used=['seasonal_rules'],
                explanation=f"Seasonal recommendation for {current_season}"
            ))
        
        return recommendations
    
    def _apply_price_rules(self, budget_profile: Dict) -> List[RecommendationScore]:
        """Apply price-based rules"""
        recommendations = []
        
        avg_budget = budget_profile.get('average_budget', 100)
        spending_tier = budget_profile.get('spending_tier', 'moderate')
        
        # Luxury recommendations for high spenders
        if spending_tier == 'luxury':
            recommendations.append(RecommendationScore(
                algorithm='knowledge_based',
                score=0.8,
                confidence=0.9,
                features_used=['price_rules', 'spending_tier'],
                explanation="Premium products matching your spending pattern"
            ))
        
        # Budget recommendations
        elif spending_tier == 'budget':
            recommendations.append(RecommendationScore(
                algorithm='knowledge_based',
                score=0.7,
                confidence=0.8,
                features_used=['price_rules', 'budget_optimization'],
                explanation="Value products within your budget range"
            ))
        
        return recommendations
    
    def _apply_temporal_rules(self, user_profile: Dict) -> List[RecommendationScore]:
        """Apply time-based rules"""
        recommendations = []
        
        temporal_patterns = user_profile.get('temporal_patterns', {})
        current_hour = datetime.now().hour
        
        # Time-of-day recommendations
        if current_hour < 12:  # Morning
            recommendations.append(RecommendationScore(
                algorithm='knowledge_based',
                score=0.6,
                confidence=0.7,
                features_used=['temporal_rules', 'time_of_day'],
                explanation="Morning shopping recommendations"
            ))
        
        return recommendations
    
    def _calculate_product_similarity(self, product1: Dict, product2: Dict) -> float:
        """Calculate similarity between two products"""
        similarity = 0.0
        
        # Category similarity
        if product1.get('category') == product2.get('category'):
            similarity += 0.5
        
        # Price similarity
        price1 = product1.get('price', 0)
        price2 = product2.get('price', 0)
        if price1 > 0 and price2 > 0:
            price_ratio = min(price1, price2) / max(price1, price2)
            similarity += price_ratio * 0.3
        
        # Name similarity (simplified)
        name1 = product1.get('name', '').lower()
        name2 = product2.get('name', '').lower()
        common_words = set(name1.split()) & set(name2.split())
        if name1 and name2:
            word_similarity = len(common_words) / max(len(name1.split()), len(name2.split()))
            similarity += word_similarity * 0.2
        
        return similarity
    
    def _get_current_season(self) -> str:
        """Get current season"""
        month = datetime.now().month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'

class EnsembleRecommender:
    """
    Ensemble recommendation system combining multiple algorithms
    """
    
    def __init__(self):
        self.content_based = ContentBasedRecommender()
        self.collaborative = CollaborativeFilteringRecommender() 
        self.matrix_factorization = MatrixFactorizationRecommender()
        self.knowledge_based = KnowledgeBasedRecommender()
        
        self.weights = {
            'content_based': 0.25,
            'collaborative_filtering': 0.30,
            'matrix_factorization': 0.25,
            'knowledge_based': 0.20
        }
        
        self.is_fitted = False
        
    def fit(self, products_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """Fit all recommendation algorithms"""
        logger.info("Fitting ensemble recommendation system...")
        
        # Fit individual recommenders
        self.content_based.fit(products_df)
        
        if not interactions_df.empty:
            self.collaborative.fit(interactions_df)
            self.matrix_factorization.fit(interactions_df)
        
        self.knowledge_based.fit(products_df)
        
        self.is_fitted = True
        logger.info("Ensemble recommendation system fitted successfully")
    
    def get_recommendations(self, user_id: int, user_profile: Dict[str, Any], 
                          num_recommendations: int = 10) -> List[ProductRecommendation]:
        """Get ensemble recommendations"""
        if not self.is_fitted:
            logger.warning("Ensemble recommender not fitted")
            return []
        
        # Get recommendations from each algorithm
        all_scores = {}
        
        # Content-based recommendations
        try:
            content_scores = self.content_based.get_recommendations(user_profile, num_recommendations * 2)
            for i, score in enumerate(content_scores):
                item_key = f"item_{i}"  # Simplified item identification
                if item_key not in all_scores:
                    all_scores[item_key] = []
                all_scores[item_key].append(score)
        except Exception as e:
            logger.error(f"Content-based recommendation error: {e}")
        
        # Collaborative filtering recommendations
        try:
            collab_scores = self.collaborative.get_recommendations(user_id, num_recommendations * 2)
            for i, score in enumerate(collab_scores):
                item_key = f"item_{i}"
                if item_key not in all_scores:
                    all_scores[item_key] = []
                all_scores[item_key].append(score)
        except Exception as e:
            logger.error(f"Collaborative filtering recommendation error: {e}")
        
        # Matrix factorization recommendations
        try:
            mf_scores = self.matrix_factorization.get_recommendations(user_id, num_recommendations * 2)
            for i, score in enumerate(mf_scores):
                item_key = f"item_{i}"
                if item_key not in all_scores:
                    all_scores[item_key] = []
                all_scores[item_key].append(score)
        except Exception as e:
            logger.error(f"Matrix factorization recommendation error: {e}")
        
        # Knowledge-based recommendations
        try:
            kb_scores = self.knowledge_based.get_recommendations(user_profile, num_recommendations * 2)
            for i, score in enumerate(kb_scores):
                item_key = f"item_{i}"
                if item_key not in all_scores:
                    all_scores[item_key] = []
                all_scores[item_key].append(score)
        except Exception as e:
            logger.error(f"Knowledge-based recommendation error: {e}")
        
        # Combine scores using weighted ensemble
        final_recommendations = []
        
        for item_key, scores in all_scores.items():
            final_score = self._calculate_ensemble_score(scores)
            confidence = self._calculate_confidence(scores)
            
            if final_score > 0.1:  # Minimum threshold
                # Generate synthetic product data (in real implementation, would fetch from database)
                product_rec = ProductRecommendation(
                    product_id=item_key,
                    product_name=f"Recommended Product {item_key}",
                    category=self._infer_category_from_scores(scores),
                    estimated_price=self._estimate_price_from_profile(user_profile),
                    platform="Multiple",
                    scores=scores,
                    final_score=final_score,
                    confidence=confidence,
                    reasons=self._generate_reasons(scores),
                    similarity_items=[]
                )
                
                final_recommendations.append(product_rec)
        
        # Sort by final score and return top recommendations
        final_recommendations.sort(key=lambda x: x.final_score, reverse=True)
        return final_recommendations[:num_recommendations]
    
    def _calculate_ensemble_score(self, scores: List[RecommendationScore]) -> float:
        """Calculate weighted ensemble score"""
        total_score = 0.0
        total_weight = 0.0
        
        for score in scores:
            weight = self.weights.get(score.algorithm, 0.1)
            total_score += score.score * weight * score.confidence
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_confidence(self, scores: List[RecommendationScore]) -> float:
        """Calculate overall confidence"""
        if not scores:
            return 0.0
        
        # Average confidence weighted by algorithm diversity
        confidence_sum = sum(score.confidence for score in scores)
        algorithm_diversity = len(set(score.algorithm for score in scores)) / len(self.weights)
        
        return (confidence_sum / len(scores)) * algorithm_diversity
    
    def _infer_category_from_scores(self, scores: List[RecommendationScore]) -> str:
        """Infer product category from recommendation scores"""
        # Simplified category inference
        return "electronics"  # Default category
    
    def _estimate_price_from_profile(self, user_profile: Dict[str, Any]) -> float:
        """Estimate price based on user profile"""
        budget_profile = user_profile.get('budget_profile', {})
        return budget_profile.get('average_budget', 100.0)
    
    def _generate_reasons(self, scores: List[RecommendationScore]) -> List[str]:
        """Generate explanation reasons from scores"""
        reasons = []
        for score in scores:
            if score.score > 0.5:
                reasons.append(score.explanation)
        
        return reasons[:3]  # Top 3 reasons
    
    def update_weights(self, performance_feedback: Dict[str, float]):
        """Update algorithm weights based on performance feedback"""
        total_performance = sum(performance_feedback.values())
        
        if total_performance > 0:
            for algorithm, performance in performance_feedback.items():
                if algorithm in self.weights:
                    # Adjust weights based on performance
                    normalized_performance = performance / total_performance
                    self.weights[algorithm] = normalized_performance
            
            # Normalize weights to sum to 1
            weight_sum = sum(self.weights.values())
            if weight_sum > 0:
                for algorithm in self.weights:
                    self.weights[algorithm] /= weight_sum
            
            logger.info(f"Updated ensemble weights: {self.weights}")

class RecommendationEvaluator:
    """
    Evaluates recommendation quality and performance
    """
    
    def __init__(self):
        self.metrics_history = []
        
    def evaluate_recommendations(self, recommendations: List[ProductRecommendation], 
                                user_feedback: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate recommendation quality"""
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'diversity': 0.0,
            'novelty': 0.0,
            'coverage': 0.0
        }
        
        if not recommendations:
            return metrics
        
        # Calculate precision (simplified)
        positive_feedback = user_feedback.get('positive_items', [])
        if positive_feedback and recommendations:
            relevant_recommendations = sum(1 for rec in recommendations 
                                         if rec.product_name in positive_feedback)
            metrics['precision'] = relevant_recommendations / len(recommendations)
        
        # Calculate diversity
        categories = [rec.category for rec in recommendations]
        unique_categories = len(set(categories))
        metrics['diversity'] = unique_categories / len(categories) if categories else 0
        
        # Calculate novelty (simplified)
        user_history = user_feedback.get('user_history', [])
        novel_items = sum(1 for rec in recommendations 
                         if rec.product_name not in user_history)
        metrics['novelty'] = novel_items / len(recommendations) if recommendations else 0
        
        # Store metrics
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        return metrics
    
    def get_performance_trends(self) -> Dict[str, List[float]]:
        """Get performance trends over time"""
        trends = defaultdict(list)
        
        for entry in self.metrics_history:
            for metric, value in entry['metrics'].items():
                trends[metric].append(value)
        
        return dict(trends)

# Flask route integration
def create_advanced_recommendation_routes(app, db):
    """Create advanced recommendation routes for Flask app"""
    ensemble_recommender = EnsembleRecommender()
    evaluator = RecommendationEvaluator()
    
    # Initialize with existing data
    try:
        # Load data and fit models (simplified)
        products_df = pd.DataFrame()  # Would load from database
        interactions_df = pd.DataFrame()  # Would load from database
        ensemble_recommender.fit(products_df, interactions_df)
    except Exception as e:
        logger.error(f"Failed to initialize ensemble recommender: {e}")
    
    @app.route('/api/recommendations/advanced/<int:user_id>', methods=['GET'])
    def get_advanced_recommendations(user_id):
        """Get advanced ensemble recommendations"""
        try:
            from flask import request
            from ml_engine import UserProfiler
            
            num_recs = request.args.get('limit', 10, type=int)
            
            # Build user profile
            profiler = UserProfiler()
            user_profile = profiler.build_user_profile(user_id)
            
            # Get ensemble recommendations
            recommendations = ensemble_recommender.get_recommendations(
                user_id, user_profile, num_recs
            )
            
            # Convert to serializable format
            recs_data = []
            for rec in recommendations:
                recs_data.append({
                    'product_id': rec.product_id,
                    'product_name': rec.product_name,
                    'category': rec.category,
                    'estimated_price': rec.estimated_price,
                    'platform': rec.platform,
                    'final_score': rec.final_score,
                    'confidence': rec.confidence,
                    'reasons': rec.reasons,
                    'algorithm_scores': [
                        {
                            'algorithm': score.algorithm,
                            'score': score.score,
                            'confidence': score.confidence,
                            'explanation': score.explanation
                        }
                        for score in rec.scores
                    ]
                })
            
            return {
                'success': True,
                'recommendations': recs_data,
                'ensemble_weights': ensemble_recommender.weights,
                'total_count': len(recs_data)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500
    
    @app.route('/api/recommendations/evaluate', methods=['POST'])
    def evaluate_recommendations():
        """Evaluate recommendation performance"""
        try:
            from flask import request
            data = request.get_json()
            
            recommendations_data = data.get('recommendations', [])
            user_feedback = data.get('feedback', {})
            
            # Convert data to ProductRecommendation objects (simplified)
            recommendations = []
            for rec_data in recommendations_data:
                rec = ProductRecommendation(
                    product_id=rec_data.get('product_id', ''),
                    product_name=rec_data.get('product_name', ''),
                    category=rec_data.get('category', ''),
                    estimated_price=rec_data.get('estimated_price', 0),
                    platform=rec_data.get('platform', ''),
                    scores=[],
                    final_score=rec_data.get('final_score', 0),
                    confidence=rec_data.get('confidence', 0),
                    reasons=rec_data.get('reasons', []),
                    similarity_items=[]
                )
                recommendations.append(rec)
            
            # Evaluate recommendations
            metrics = evaluator.evaluate_recommendations(recommendations, user_feedback)
            
            return {
                'success': True,
                'evaluation_metrics': metrics,
                'performance_trends': evaluator.get_performance_trends()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500
    
    @app.route('/api/recommendations/update-weights', methods=['POST'])
    def update_ensemble_weights():
        """Update ensemble algorithm weights"""
        try:
            from flask import request
            data = request.get_json()
            
            performance_feedback = data.get('performance_feedback', {})
            ensemble_recommender.update_weights(performance_feedback)
            
            return {
                'success': True,
                'updated_weights': ensemble_recommender.weights,
                'message': 'Ensemble weights updated successfully'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500

if __name__ == "__main__":
    # Test the advanced recommendation system
    ensemble = EnsembleRecommender()
    
    # Create sample data for testing
    products_df = pd.DataFrame({
        'id': range(1, 6),
        'name': ['Laptop', 'Phone', 'Headphones', 'Tablet', 'Monitor'],
        'category': ['electronics', 'electronics', 'electronics', 'electronics', 'electronics'],
        'price': [999, 599, 199, 399, 299]
    })
    
    interactions_df = pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3],
        'product_name': ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Monitor'],
        'rating': [5, 4, 3, 5, 4]
    })
    
    # Fit and test
    ensemble.fit(products_df, interactions_df)
    
    user_profile = {
        'preferences': {'preferred_terms': ['laptop', 'computer']},
        'category_preferences': {'electronics': 0.8},
        'budget_profile': {'average_budget': 800}
    }
    
    recommendations = ensemble.get_recommendations(1, user_profile, 5)
    print(f"Generated {len(recommendations)} advanced recommendations")
    
    for rec in recommendations:
        print(f"- {rec.product_name} (Score: {rec.final_score:.2f}, Confidence: {rec.confidence:.2f})")
