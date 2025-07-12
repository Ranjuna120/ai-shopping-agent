"""
Advanced Analytics Module for AI Shopping Agent
Provides comprehensive analytics, insights, and reporting capabilities
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
import statistics
from typing import Dict, List, Tuple, Optional, Any
import re
from dataclasses import dataclass
from flask import jsonify

@dataclass
class AnalyticsReport:
    """Data class for analytics reports"""
    report_type: str
    generated_at: datetime
    data: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]

class ShoppingAnalytics:
    """
    Comprehensive analytics engine for shopping behavior analysis
    """
    
    def __init__(self, db_path: str = 'shopping_agent.db'):
        self.db_path = db_path
        self.insights_cache = {}
        self.last_cache_update = None
        
    def get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def get_user_behavior_analytics(self, user_id: int, days: int = 30) -> AnalyticsReport:
        """
        Analyze user shopping behavior patterns
        """
        conn = self.get_db_connection()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get user search history
        search_query = """
        SELECT * FROM search_history 
        WHERE user_id = ? AND created_at >= ? AND created_at <= ?
        ORDER BY created_at DESC
        """
        
        try:
            searches = pd.read_sql_query(
                search_query, 
                conn, 
                params=[user_id, start_date.isoformat(), end_date.isoformat()]
            )
        except Exception as e:
            searches = pd.DataFrame()  # Return empty if table doesn't exist
        
        # Get favorites data
        favorites_query = """
        SELECT * FROM favorites 
        WHERE user_id = ? AND added_at >= ? AND added_at <= ?
        ORDER BY added_at DESC
        """
        
        try:
            favorites = pd.read_sql_query(
                favorites_query,
                conn,
                params=[user_id, start_date.isoformat(), end_date.isoformat()]
            )
        except Exception as e:
            favorites = pd.DataFrame()  # Return empty if table doesn't exist
        
        # Get price alerts data
        alerts_query = """
        SELECT * FROM price_alert 
        WHERE user_id = ? AND created_at >= ? AND created_at <= ?
        ORDER BY created_at DESC
        """
        
        try:
            alerts = pd.read_sql_query(
                alerts_query,
                conn,
                params=[user_id, start_date.isoformat(), end_date.isoformat()]
            )
        except Exception as e:
            alerts = pd.DataFrame()  # Return empty if table doesn't exist
        
        conn.close()
        
        # Analyze data
        analytics_data = self._analyze_user_behavior(searches, favorites, alerts)
        insights = self._generate_user_insights(analytics_data)
        recommendations = self._generate_user_recommendations(analytics_data)
        
        return AnalyticsReport(
            report_type="user_behavior",
            generated_at=datetime.now(),
            data=analytics_data,
            insights=insights,
            recommendations=recommendations
        )
    
    def _analyze_user_behavior(self, searches: pd.DataFrame, favorites: pd.DataFrame, alerts: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform detailed analysis of user behavior data
        """
        analytics = {
            'search_patterns': self._analyze_search_patterns(searches),
            'favorites_analysis': self._analyze_favorites(favorites),
            'price_sensitivity': self._analyze_price_sensitivity(alerts, favorites),
            'platform_preferences': self._analyze_platform_preferences(favorites),
            'temporal_patterns': self._analyze_temporal_patterns(searches, favorites, alerts),
            'budget_analysis': self._analyze_budget_patterns(searches),
            'category_interests': self._analyze_category_interests(searches, favorites)
        }
        
        return analytics
    
    def _analyze_search_patterns(self, searches: pd.DataFrame) -> Dict[str, Any]:
        """Analyze search behavior patterns"""
        if searches.empty:
            return {'total_searches': 0, 'patterns': {}}
        
        # Extract search terms and analyze frequency
        search_terms = []
        budgets = []
        
        for _, search in searches.iterrows():
            if hasattr(search, 'query') and search.query:
                terms = self._extract_keywords(search.query)
                search_terms.extend(terms)
            
            if hasattr(search, 'budget') and search.budget:
                try:
                    budgets.append(float(search.budget))
                except (ValueError, TypeError):
                    pass
        
        # Analyze search frequency and patterns
        term_frequency = Counter(search_terms)
        
        patterns = {
            'total_searches': len(searches),
            'unique_terms': len(set(search_terms)),
            'top_search_terms': dict(term_frequency.most_common(10)),
            'search_frequency': len(searches) / 30 if len(searches) > 0 else 0,
            'average_budget': np.mean(budgets) if budgets else 0,
            'budget_range': {
                'min': min(budgets) if budgets else 0,
                'max': max(budgets) if budgets else 0,
                'std': np.std(budgets) if len(budgets) > 1 else 0
            }
        }
        
        return patterns
    
    def _analyze_favorites(self, favorites: pd.DataFrame) -> Dict[str, Any]:
        """Analyze favorites patterns"""
        if favorites.empty:
            return {'total_favorites': 0, 'analysis': {}}
        
        # Extract price data
        prices = []
        platforms = []
        categories = []
        
        for _, fav in favorites.iterrows():
            if hasattr(fav, 'price') and fav.price:
                try:
                    prices.append(float(fav.price))
                except (ValueError, TypeError):
                    pass
            
            if hasattr(fav, 'platform') and fav.platform:
                platforms.append(fav.platform.lower())
            
            if hasattr(fav, 'product_name') and fav.product_name:
                category = self._categorize_product(fav.product_name)
                categories.append(category)
        
        analysis = {
            'total_favorites': len(favorites),
            'price_analysis': {
                'average_price': np.mean(prices) if prices else 0,
                'price_range': {
                    'min': min(prices) if prices else 0,
                    'max': max(prices) if prices else 0,
                    'median': np.median(prices) if prices else 0
                },
                'price_distribution': self._create_price_distribution(prices)
            },
            'platform_distribution': dict(Counter(platforms)),
            'category_distribution': dict(Counter(categories)),
            'favorites_per_week': len(favorites) / 4.3  # approximate weeks in a month
        }
        
        return analysis
    
    def _analyze_price_sensitivity(self, alerts: pd.DataFrame, favorites: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price sensitivity and alert patterns"""
        if alerts.empty:
            return {'price_sensitivity': 'unknown', 'alert_patterns': {}}
        
        target_prices = []
        current_prices = []
        
        for _, alert in alerts.iterrows():
            if hasattr(alert, 'target_price') and alert.target_price:
                try:
                    target_prices.append(float(alert.target_price))
                except (ValueError, TypeError):
                    pass
            
            if hasattr(alert, 'current_price') and alert.current_price:
                try:
                    current_prices.append(float(alert.current_price))
                except (ValueError, TypeError):
                    pass
        
        # Calculate price sensitivity
        sensitivity_score = self._calculate_price_sensitivity(target_prices, current_prices, favorites)
        
        analysis = {
            'total_alerts': len(alerts),
            'average_target_price': np.mean(target_prices) if target_prices else 0,
            'price_sensitivity_score': sensitivity_score,
            'sensitivity_level': self._categorize_price_sensitivity(sensitivity_score),
            'alert_frequency': len(alerts) / 30,  # alerts per day
            'price_drop_expectations': {
                'average_expected_drop': np.mean([cp - tp for cp, tp in zip(current_prices, target_prices) if cp and tp]) if current_prices and target_prices else 0,
                'max_expected_drop': max([cp - tp for cp, tp in zip(current_prices, target_prices) if cp and tp]) if current_prices and target_prices else 0
            }
        }
        
        return analysis
    
    def _analyze_platform_preferences(self, favorites: pd.DataFrame) -> Dict[str, Any]:
        """Analyze platform usage preferences"""
        if favorites.empty:
            return {'platform_preferences': {}, 'diversity_score': 0}
        
        platforms = [fav.platform.lower() if hasattr(fav, 'platform') and fav.platform else 'unknown' 
                    for _, fav in favorites.iterrows()]
        
        platform_counts = Counter(platforms)
        total_platforms = len(set(platforms))
        diversity_score = total_platforms / len(platforms) if platforms else 0
        
        # Calculate platform loyalty
        most_used_platform = platform_counts.most_common(1)[0] if platform_counts else ('unknown', 0)
        loyalty_score = most_used_platform[1] / len(platforms) if platforms else 0
        
        preferences = {
            'platform_distribution': dict(platform_counts),
            'most_preferred_platform': most_used_platform[0],
            'platform_diversity_score': diversity_score,
            'platform_loyalty_score': loyalty_score,
            'total_platforms_used': total_platforms
        }
        
        return preferences
    
    def _analyze_temporal_patterns(self, searches: pd.DataFrame, favorites: pd.DataFrame, alerts: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal shopping patterns"""
        all_activities = []
        
        # Combine all activities with timestamps
        for df, activity_type in [(searches, 'search'), (favorites, 'favorite'), (alerts, 'alert')]:
            if not df.empty:
                for _, row in df.iterrows():
                    timestamp_col = 'created_at' if hasattr(row, 'created_at') else 'added_at'
                    if hasattr(row, timestamp_col) and getattr(row, timestamp_col):
                        try:
                            dt = datetime.fromisoformat(getattr(row, timestamp_col))
                            all_activities.append({
                                'datetime': dt,
                                'hour': dt.hour,
                                'day_of_week': dt.weekday(),
                                'activity_type': activity_type
                            })
                        except (ValueError, TypeError):
                            continue
        
        if not all_activities:
            return {'temporal_patterns': {}, 'peak_hours': [], 'peak_days': []}
        
        # Analyze patterns
        hours = [activity['hour'] for activity in all_activities]
        days = [activity['day_of_week'] for activity in all_activities]
        
        hour_distribution = Counter(hours)
        day_distribution = Counter(days)
        
        patterns = {
            'peak_hours': [hour for hour, _ in hour_distribution.most_common(3)],
            'peak_days': [self._day_name(day) for day, _ in day_distribution.most_common(3)],
            'hour_distribution': dict(hour_distribution),
            'day_distribution': {self._day_name(day): count for day, count in day_distribution.items()},
            'activity_timeline': self._create_activity_timeline(all_activities),
            'shopping_rhythm': self._analyze_shopping_rhythm(all_activities)
        }
        
        return patterns
    
    def _analyze_budget_patterns(self, searches: pd.DataFrame) -> Dict[str, Any]:
        """Analyze budget allocation patterns"""
        if searches.empty:
            return {'budget_patterns': {}, 'spending_profile': 'unknown'}
        
        budgets = []
        categories = []
        
        for _, search in searches.iterrows():
            if hasattr(search, 'budget') and search.budget:
                try:
                    budget = float(search.budget)
                    budgets.append(budget)
                    
                    if hasattr(search, 'query') and search.query:
                        category = self._categorize_product(search.query)
                        categories.append((budget, category))
                except (ValueError, TypeError):
                    pass
        
        if not budgets:
            return {'budget_patterns': {}, 'spending_profile': 'no_data'}
        
        # Analyze budget patterns
        budget_analysis = {
            'average_budget': np.mean(budgets),
            'median_budget': np.median(budgets),
            'budget_range': {'min': min(budgets), 'max': max(budgets)},
            'budget_variance': np.var(budgets),
            'spending_profile': self._determine_spending_profile(budgets),
            'category_budgets': self._analyze_category_budgets(categories),
            'budget_trends': self._analyze_budget_trends(budgets)
        }
        
        return budget_analysis
    
    def _analyze_category_interests(self, searches: pd.DataFrame, favorites: pd.DataFrame) -> Dict[str, Any]:
        """Analyze product category interests"""
        categories = []
        
        # Extract categories from searches
        if not searches.empty:
            for _, search in searches.iterrows():
                if hasattr(search, 'query') and search.query:
                    category = self._categorize_product(search.query)
                    categories.append(category)
        
        # Extract categories from favorites
        if not favorites.empty:
            for _, fav in favorites.iterrows():
                if hasattr(fav, 'product_name') and fav.product_name:
                    category = self._categorize_product(fav.product_name)
                    categories.append(category)
        
        if not categories:
            return {'category_interests': {}, 'primary_interests': []}
        
        category_counts = Counter(categories)
        total_items = len(categories)
        
        interests = {
            'category_distribution': dict(category_counts),
            'primary_interests': [cat for cat, _ in category_counts.most_common(5)],
            'category_diversity': len(set(categories)) / total_items if total_items > 0 else 0,
            'interest_concentration': category_counts.most_common(1)[0][1] / total_items if category_counts else 0
        }
        
        return interests
    
    # Helper methods
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        if not text:
            return []
        
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _categorize_product(self, product_name: str) -> str:
        """Categorize product based on name"""
        if not product_name:
            return 'unknown'
        
        product_lower = product_name.lower()
        
        categories = {
            'electronics': ['laptop', 'computer', 'phone', 'smartphone', 'tablet', 'camera', 'headphone', 'speaker'],
            'clothing': ['shirt', 'pant', 'dress', 'shoe', 'jacket', 'coat', 'sweater'],
            'home': ['furniture', 'chair', 'table', 'bed', 'sofa', 'lamp', 'rug'],
            'sports': ['fitness', 'exercise', 'sport', 'gym', 'outdoor', 'bike'],
            'books': ['book', 'novel', 'textbook', 'kindle'],
            'beauty': ['makeup', 'skincare', 'beauty', 'cosmetic'],
            'automotive': ['car', 'auto', 'tire', 'battery'],
            'gaming': ['game', 'gaming', 'console', 'controller']
        }
        
        for category, keywords in categories.items():
            if any(keyword in product_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def _create_price_distribution(self, prices: List[float]) -> Dict[str, int]:
        """Create price distribution buckets"""
        if not prices:
            return {}
        
        buckets = {
            '0-50': 0, '51-100': 0, '101-250': 0, 
            '251-500': 0, '501-1000': 0, '1000+': 0
        }
        
        for price in prices:
            if price <= 50:
                buckets['0-50'] += 1
            elif price <= 100:
                buckets['51-100'] += 1
            elif price <= 250:
                buckets['101-250'] += 1
            elif price <= 500:
                buckets['251-500'] += 1
            elif price <= 1000:
                buckets['501-1000'] += 1
            else:
                buckets['1000+'] += 1
        
        return buckets
    
    def _calculate_price_sensitivity(self, target_prices: List[float], current_prices: List[float], favorites: pd.DataFrame) -> float:
        """Calculate price sensitivity score (0-1, where 1 is most sensitive)"""
        if not target_prices and not current_prices:
            return 0.5  # neutral
        
        # Calculate based on price alert patterns and favorites
        sensitivity_factors = []
        
        # Factor 1: How much discount they expect
        if target_prices and current_prices:
            discounts = [(cp - tp) / cp for cp, tp in zip(current_prices, target_prices) if cp > 0]
            if discounts:
                avg_discount_expected = np.mean(discounts)
                # Higher expected discount = higher sensitivity
                sensitivity_factors.append(min(avg_discount_expected * 2, 1.0))
        
        # Factor 2: Number of price alerts relative to favorites
        if not favorites.empty:
            alert_to_fav_ratio = len(target_prices) / len(favorites)
            sensitivity_factors.append(min(alert_to_fav_ratio, 1.0))
        
        # Factor 3: Variance in target prices (consistent expectations = higher sensitivity)
        if len(target_prices) > 1:
            price_variance = np.var(target_prices)
            normalized_variance = 1 - min(price_variance / np.mean(target_prices) if np.mean(target_prices) > 0 else 0, 1)
            sensitivity_factors.append(normalized_variance)
        
        return np.mean(sensitivity_factors) if sensitivity_factors else 0.5
    
    def _categorize_price_sensitivity(self, score: float) -> str:
        """Categorize price sensitivity score"""
        if score >= 0.7:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _day_name(self, day_num: int) -> str:
        """Convert day number to name"""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return days[day_num] if 0 <= day_num <= 6 else 'Unknown'
    
    def _create_activity_timeline(self, activities: List[Dict]) -> Dict[str, Any]:
        """Create activity timeline analysis"""
        if not activities:
            return {}
        
        # Group by date
        daily_activity = defaultdict(int)
        for activity in activities:
            date_key = activity['datetime'].strftime('%Y-%m-%d')
            daily_activity[date_key] += 1
        
        # Calculate trends
        dates = sorted(daily_activity.keys())
        activity_counts = [daily_activity[date] for date in dates]
        
        timeline = {
            'daily_activity': dict(daily_activity),
            'most_active_date': max(daily_activity.items(), key=lambda x: x[1])[0] if daily_activity else None,
            'activity_trend': 'increasing' if len(activity_counts) > 1 and activity_counts[-1] > activity_counts[0] else 'stable',
            'average_daily_activity': np.mean(list(daily_activity.values())) if daily_activity else 0
        }
        
        return timeline
    
    def _analyze_shopping_rhythm(self, activities: List[Dict]) -> Dict[str, Any]:
        """Analyze shopping rhythm and patterns"""
        if not activities:
            return {}
        
        # Analyze time gaps between activities
        activities_sorted = sorted(activities, key=lambda x: x['datetime'])
        time_gaps = []
        
        for i in range(1, len(activities_sorted)):
            gap = activities_sorted[i]['datetime'] - activities_sorted[i-1]['datetime']
            time_gaps.append(gap.total_seconds() / 3600)  # in hours
        
        rhythm = {
            'average_gap_hours': np.mean(time_gaps) if time_gaps else 0,
            'shopping_frequency': len(activities) / 30,  # per day
            'burst_shopping': len([gap for gap in time_gaps if gap < 1]) / len(time_gaps) if time_gaps else 0,  # % of activities within 1 hour
            'rhythm_type': self._determine_rhythm_type(time_gaps)
        }
        
        return rhythm
    
    def _analyze_category_budgets(self, category_budgets: List[Tuple[float, str]]) -> Dict[str, Dict[str, float]]:
        """Analyze budget allocation per category"""
        if not category_budgets:
            return {}
        
        category_data = defaultdict(list)
        for budget, category in category_budgets:
            category_data[category].append(budget)
        
        analysis = {}
        for category, budgets in category_data.items():
            analysis[category] = {
                'average_budget': np.mean(budgets),
                'total_searches': len(budgets),
                'budget_range': {'min': min(budgets), 'max': max(budgets)}
            }
        
        return analysis
    
    def _analyze_budget_trends(self, budgets: List[float]) -> Dict[str, Any]:
        """Analyze budget trends over time"""
        if len(budgets) < 2:
            return {'trend': 'insufficient_data'}
        
        # Simple trend analysis
        first_half = budgets[:len(budgets)//2]
        second_half = budgets[len(budgets)//2:]
        
        first_avg = np.mean(first_half)
        second_avg = np.mean(second_half)
        
        trend = {
            'direction': 'increasing' if second_avg > first_avg else 'decreasing',
            'change_percentage': ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0,
            'budget_stability': 'stable' if abs(second_avg - first_avg) / first_avg < 0.1 else 'variable'
        }
        
        return trend
    
    def _determine_spending_profile(self, budgets: List[float]) -> str:
        """Determine spending profile based on budget patterns"""
        if not budgets:
            return 'unknown'
        
        avg_budget = np.mean(budgets)
        budget_std = np.std(budgets)
        
        if avg_budget < 100:
            if budget_std < avg_budget * 0.3:
                return 'consistent_budget_shopper'
            else:
                return 'variable_budget_shopper'
        elif avg_budget < 500:
            if budget_std < avg_budget * 0.3:
                return 'consistent_mid_range_shopper'
            else:
                return 'variable_mid_range_shopper'
        else:
            if budget_std < avg_budget * 0.3:
                return 'consistent_premium_shopper'
            else:
                return 'variable_premium_shopper'
    
    def _determine_rhythm_type(self, time_gaps: List[float]) -> str:
        """Determine shopping rhythm type"""
        if not time_gaps:
            return 'unknown'
        
        avg_gap = np.mean(time_gaps)
        gap_std = np.std(time_gaps)
        
        if avg_gap < 24:  # Less than 24 hours
            return 'frequent_shopper'
        elif avg_gap < 168:  # Less than a week
            return 'regular_shopper'
        else:
            return 'occasional_shopper'
    
    def _generate_user_insights(self, analytics_data: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from analytics data"""
        insights = []
        
        # Search pattern insights
        search_patterns = analytics_data.get('search_patterns', {})
        if search_patterns.get('search_frequency', 0) > 1:
            insights.append(f"You search for products {search_patterns['search_frequency']:.1f} times per day, showing active shopping behavior.")
        
        # Budget insights
        budget_analysis = analytics_data.get('budget_analysis', {})
        if budget_analysis.get('average_budget', 0) > 0:
            avg_budget = budget_analysis['average_budget']
            insights.append(f"Your average shopping budget is ${avg_budget:.2f}, indicating {budget_analysis.get('spending_profile', 'unknown')} spending behavior.")
        
        # Price sensitivity insights
        price_sensitivity = analytics_data.get('price_sensitivity', {})
        sensitivity_level = price_sensitivity.get('sensitivity_level', 'unknown')
        if sensitivity_level != 'unknown':
            insights.append(f"You have {sensitivity_level} price sensitivity, with {price_sensitivity.get('total_alerts', 0)} active price alerts.")
        
        # Platform preferences
        platform_prefs = analytics_data.get('platform_preferences', {})
        most_preferred = platform_prefs.get('most_preferred_platform', 'unknown')
        if most_preferred != 'unknown':
            insights.append(f"You prefer shopping on {most_preferred.title()}, showing platform loyalty.")
        
        # Category interests
        category_interests = analytics_data.get('category_interests', {})
        primary_interests = category_interests.get('primary_interests', [])
        if primary_interests:
            insights.append(f"Your top interests are: {', '.join(primary_interests[:3])}.")
        
        return insights
    
    def _generate_user_recommendations(self, analytics_data: Dict[str, Any]) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Budget recommendations
        budget_analysis = analytics_data.get('budget_analysis', {})
        spending_profile = budget_analysis.get('spending_profile', '')
        if 'variable' in spending_profile:
            recommendations.append("Consider setting a consistent monthly shopping budget to better track your expenses.")
        
        # Price sensitivity recommendations
        price_sensitivity = analytics_data.get('price_sensitivity', {})
        if price_sensitivity.get('sensitivity_level') == 'high':
            recommendations.append("Set up more price alerts for products you're interested in to maximize savings.")
        elif price_sensitivity.get('sensitivity_level') == 'low':
            recommendations.append("Consider using price comparison tools to find better deals.")
        
        # Platform diversity recommendations
        platform_prefs = analytics_data.get('platform_preferences', {})
        diversity_score = platform_prefs.get('platform_diversity_score', 0)
        if diversity_score < 0.3:
            recommendations.append("Try exploring different shopping platforms to find better deals and variety.")
        
        # Category recommendations
        category_interests = analytics_data.get('category_interests', {})
        if category_interests.get('category_diversity', 0) < 0.3:
            recommendations.append("Explore different product categories to discover new interests and needs.")
        
        # Temporal pattern recommendations
        temporal_patterns = analytics_data.get('temporal_patterns', {})
        peak_hours = temporal_patterns.get('peak_hours', [])
        if peak_hours and max(peak_hours) < 6:  # Early morning shopping
            recommendations.append("You tend to shop early in the morning - this is great for catching flash sales and new deals!")
        
        return recommendations

class ReportGenerator:
    """
    Generate comprehensive reports and visualizations
    """
    
    def __init__(self, analytics: ShoppingAnalytics):
        self.analytics = analytics
    
    def generate_comprehensive_report(self, user_id: int, report_type: str = 'full') -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        
        if report_type == 'full':
            # Generate full 30-day report
            behavior_report = self.analytics.get_user_behavior_analytics(user_id, 30)
            
            report = {
                'report_id': f"report_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'user_id': user_id,
                'report_type': 'comprehensive_30_day',
                'generated_at': datetime.now().isoformat(),
                'summary': self._generate_summary(behavior_report),
                'detailed_analytics': behavior_report.data,
                'insights': behavior_report.insights,
                'recommendations': behavior_report.recommendations,
                'score_card': self._generate_score_card(behavior_report.data)
            }
            
        return report
    
    def _generate_summary(self, report: AnalyticsReport) -> Dict[str, Any]:
        """Generate executive summary"""
        data = report.data
        
        summary = {
            'total_activities': (
                data.get('search_patterns', {}).get('total_searches', 0) +
                data.get('favorites_analysis', {}).get('total_favorites', 0) +
                data.get('price_sensitivity', {}).get('total_alerts', 0)
            ),
            'engagement_level': self._calculate_engagement_level(data),
            'primary_shopping_profile': self._determine_primary_profile(data),
            'key_metrics': {
                'searches_per_day': data.get('search_patterns', {}).get('search_frequency', 0),
                'average_budget': data.get('budget_analysis', {}).get('average_budget', 0),
                'price_sensitivity': data.get('price_sensitivity', {}).get('sensitivity_level', 'unknown'),
                'platform_loyalty': data.get('platform_preferences', {}).get('platform_loyalty_score', 0)
            }
        }
        
        return summary
    
    def _calculate_engagement_level(self, data: Dict[str, Any]) -> str:
        """Calculate user engagement level"""
        search_freq = data.get('search_patterns', {}).get('search_frequency', 0)
        total_favorites = data.get('favorites_analysis', {}).get('total_favorites', 0)
        total_alerts = data.get('price_sensitivity', {}).get('total_alerts', 0)
        
        engagement_score = search_freq * 0.4 + (total_favorites / 30) * 0.3 + (total_alerts / 30) * 0.3
        
        if engagement_score >= 1.5:
            return 'high'
        elif engagement_score >= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _determine_primary_profile(self, data: Dict[str, Any]) -> str:
        """Determine primary shopping profile"""
        spending_profile = data.get('budget_analysis', {}).get('spending_profile', '')
        price_sensitivity = data.get('price_sensitivity', {}).get('sensitivity_level', 'unknown')
        platform_loyalty = data.get('platform_preferences', {}).get('platform_loyalty_score', 0)
        
        profiles = []
        
        if 'premium' in spending_profile:
            profiles.append('premium_shopper')
        elif 'budget' in spending_profile:
            profiles.append('budget_conscious')
        
        if price_sensitivity == 'high':
            profiles.append('deal_hunter')
        
        if platform_loyalty > 0.7:
            profiles.append('brand_loyal')
        
        return '_'.join(profiles) if profiles else 'casual_shopper'
    
    def _generate_score_card(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Generate shopping behavior score card"""
        scores = {
            'activity_score': min(data.get('search_patterns', {}).get('search_frequency', 0) * 10, 100),
            'budget_consistency': self._calculate_budget_consistency_score(data),
            'price_awareness': self._calculate_price_awareness_score(data),
            'platform_exploration': data.get('platform_preferences', {}).get('platform_diversity_score', 0) * 100,
            'category_diversity': data.get('category_interests', {}).get('category_diversity', 0) * 100
        }
        
        scores['overall_score'] = np.mean(list(scores.values()))
        
        return scores
    
    def _calculate_budget_consistency_score(self, data: Dict[str, Any]) -> float:
        """Calculate budget consistency score"""
        budget_data = data.get('budget_analysis', {})
        spending_profile = budget_data.get('spending_profile', '')
        
        if 'consistent' in spending_profile:
            return 85.0
        elif 'variable' in spending_profile:
            return 45.0
        else:
            return 50.0
    
    def _calculate_price_awareness_score(self, data: Dict[str, Any]) -> float:
        """Calculate price awareness score"""
        price_data = data.get('price_sensitivity', {})
        sensitivity_level = price_data.get('sensitivity_level', 'unknown')
        total_alerts = price_data.get('total_alerts', 0)
        
        base_score = {'high': 90, 'medium': 70, 'low': 40, 'unknown': 50}.get(sensitivity_level, 50)
        alert_bonus = min(total_alerts * 5, 20)  # Up to 20 point bonus for alerts
        
        return min(base_score + alert_bonus, 100)

# Flask route integration functions
def create_analytics_routes(app, db):
    """Create analytics routes for the Flask app"""
    analytics = ShoppingAnalytics()
    report_generator = ReportGenerator(analytics)
    
    @app.route('/api/analytics/user/<int:user_id>', methods=['GET'])
    def get_user_analytics(user_id):
        """Get user analytics report"""
        try:
            days = request.args.get('days', 30, type=int)
            report = analytics.get_user_behavior_analytics(user_id, days)
            
            return jsonify({
                'success': True,
                'report': {
                    'type': report.report_type,
                    'generated_at': report.generated_at.isoformat(),
                    'data': report.data,
                    'insights': report.insights,
                    'recommendations': report.recommendations
                }
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/analytics/report/<int:user_id>', methods=['GET'])
    def get_comprehensive_report(user_id):
        """Get comprehensive analytics report"""
        try:
            report_type = request.args.get('type', 'full')
            report = report_generator.generate_comprehensive_report(user_id, report_type)
            
            return jsonify({
                'success': True,
                'report': report
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == "__main__":
    # Test the analytics module
    analytics = ShoppingAnalytics()
    report = analytics.get_user_behavior_analytics(1, 30)
    print("Analytics test completed successfully!")
