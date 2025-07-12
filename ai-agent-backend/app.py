from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///shopping_agent.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Import models and initialize database
from models import db, User, UserPreference, SearchHistory, Favorite, PriceAlert

# Initialize database with app
db.init_app(app)

# Import auth and other managers
from auth import AuthManager
from price_alerts import PriceAlertManager
from enhanced_scraper import EnhancedScraper
from analytics import create_analytics_routes
from ml_engine import create_ml_routes  
from email_service import create_email_routes
from data_processing import create_data_processing_routes
from advanced_recommendations import create_advanced_recommendation_routes

# Initialize managers
auth_manager = AuthManager(app, db)
price_alert_manager = PriceAlertManager(app)
scraper = EnhancedScraper()

# Register authentication routes
auth_manager.register_routes()
price_alert_manager.register_routes(auth_manager)

# Register additional module routes
create_analytics_routes(app, db)
create_ml_routes(app, db)
create_email_routes(app, db)
create_data_processing_routes(app, db)
create_advanced_recommendation_routes(app, db)

# Create database tables
with app.app_context():
    db.create_all()

# Basic product recommendation endpoint
@app.route("/api/auth/recommend", methods=["POST"])
@auth_manager.require_auth
def recommend():
    try:
        data = request.json
        budget = float(data.get("budget", 0))
        product_type = data.get("product", "").lower()
        use_nlp = data.get("use_nlp", False)
        currency = data.get("currency", "USD")

        if not product_type or budget <= 0:
            return jsonify({"error": "Please provide valid product and budget"}), 400

        # Save search to history
        search_history = SearchHistory(
            user_id=request.current_user_id,
            query=product_type,
            budget=budget
        )
        db.session.add(search_history)
        db.session.commit()

        # Use enhanced scraper to get products
        products = scraper.scrape_all_platforms(product_type, max_results_per_platform=5)
        
        # Filter by budget
        filtered_products = [
            product for product in products 
            if product.get("price", 0) <= budget
        ]

        # If we have very few results, add some sample/demo products
        if len(filtered_products) < 3:
            sample_products = [
                {
                    "name": f"Sample {product_type.title()} - Basic Model",
                    "price": budget * 0.6,
                    "rating": 4.2,
                    "image": "",
                    "url": "https://example.com/product1",
                    "platform": "Demo Store"
                },
                {
                    "name": f"Premium {product_type.title()} - Advanced Features",
                    "price": budget * 0.8,
                    "rating": 4.5,
                    "image": "",
                    "url": "https://example.com/product2", 
                    "platform": "Demo Store"
                },
                {
                    "name": f"Budget {product_type.title()} - Great Value",
                    "price": budget * 0.4,
                    "rating": 4.0,
                    "image": "",
                    "url": "https://example.com/product3",
                    "platform": "Demo Store"
                }
            ]
            
            # Add sample products that fit the budget
            for sample in sample_products:
                if sample["price"] <= budget:
                    filtered_products.append(sample)

        # Update search history with results count
        search_history.results_count = len(filtered_products)
        db.session.commit()

        # Sort by price
        filtered_products.sort(key=lambda x: x.get("price", 0))

        return jsonify(filtered_products[:15])  # Return top 15 results

    except Exception as e:
        print(f"Error in recommend: {e}")
        return jsonify({"error": "Search failed. Please try again."}), 500

# Get user's search history
@app.route("/api/auth/search-history", methods=["GET"])
@auth_manager.require_auth
def get_search_history():
    try:
        history = SearchHistory.query.filter_by(
            user_id=request.current_user_id
        ).order_by(SearchHistory.timestamp.desc()).limit(10).all()
        
        return jsonify([{
            'id': h.id,
            'query': h.query,
            'budget': h.budget,
            'results_count': h.results_count,
            'timestamp': h.timestamp.isoformat()
        } for h in history])
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add product to favorites
@app.route("/api/favorites", methods=["POST"])
@auth_manager.require_auth
def add_favorite():
    try:
        data = request.json
        
        # Check if already in favorites
        existing = Favorite.query.filter_by(
            user_id=request.current_user_id,
            product_url=data['product_url']
        ).first()
        
        if existing:
            return jsonify({"message": "Already in favorites"}), 200
        
        favorite = Favorite(
            user_id=request.current_user_id,
            product_name=data['product_name'],
            product_url=data['product_url'],
            price=data.get('price'),
            platform=data.get('platform')
        )
        
        db.session.add(favorite)
        db.session.commit()
        
        return jsonify({"message": "Added to favorites successfully"}), 201
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Get user's favorites
@app.route("/api/favorites", methods=["GET"])
@auth_manager.require_auth
def get_favorites():
    try:
        favorites = Favorite.query.filter_by(
            user_id=request.current_user_id
        ).order_by(Favorite.added_at.desc()).all()
        
        return jsonify([{
            'id': f.id,
            'product_name': f.product_name,
            'product_url': f.product_url,
            'price': f.price,
            'platform': f.platform,
            'added_at': f.added_at.isoformat()
        } for f in favorites])
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Remove from favorites
@app.route("/api/favorites/<int:favorite_id>", methods=["DELETE"])
@auth_manager.require_auth
def remove_favorite(favorite_id):
    try:
        favorite = Favorite.query.filter_by(
            id=favorite_id,
            user_id=request.current_user_id
        ).first()
        
        if not favorite:
            return jsonify({"error": "Favorite not found"}), 404
        
        db.session.delete(favorite)
        db.session.commit()
        
        return jsonify({"message": "Removed from favorites"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "AI Shopping Agent Backend is running!"
    })

# Test endpoint (no auth required)
@app.route("/api/test", methods=["GET"])
def test():
    return jsonify({
        "message": "Backend is working!",
        "available_endpoints": [
            "/api/auth/register",
            "/api/auth/login", 
            "/api/auth/profile",
            "/api/auth/recommend",
            "/api/favorites",
            "/api/price-alerts"
        ]
    })

if __name__ == "__main__":
    print("🚀 Starting AI Shopping Agent Backend...")
    print("📡 Available at: http://127.0.0.1:5000")
    print("🔗 Test endpoint: http://127.0.0.1:5000/api/test")
    app.run(debug=True, host='127.0.0.1', port=5000)
