from flask import request, jsonify
from models import PriceAlert, db
from enhanced_scraper import EnhancedScraper
import threading
import time
import smtplib
from email.mime.text import MIMEText
import os

class PriceAlertManager:
    def __init__(self, app):
        self.app = app
        self.scraper = EnhancedScraper()
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', 587)),
            'email': os.getenv('EMAIL_USER'),
            'password': os.getenv('EMAIL_PASSWORD')
        }
        
        # Start background price monitoring
        self.start_price_monitoring()
    
    def send_email_alert(self, user_email, product_name, target_price, current_price, product_url):
        """Send email notification when price drops"""
        try:
            msg = MIMEText(f"""
            Great news! The price for "{product_name}" has dropped!
            
            Target Price: ${target_price:.2f}
            Current Price: ${current_price:.2f}
            Savings: ${target_price - current_price:.2f}
            
            View Product: {product_url}
            
            Happy Shopping!
            AI Shopping Agent
            """)
            
            msg['Subject'] = f'Price Alert: {product_name} is now ${current_price:.2f}!'
            msg['From'] = self.email_config['email']
            msg['To'] = user_email
            
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['email'], self.email_config['password'])
                server.send_message(msg)
                
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def check_price_alerts(self):
        """Check all active price alerts"""
        with self.app.app_context():
            alerts = PriceAlert.query.filter_by(is_active=True).all()
            
            for alert in alerts:
                try:
                    # Get current price by scraping the product URL
                    current_price = self.get_current_price(alert.product_url)
                    
                    if current_price and current_price <= alert.target_price:
                        # Price target reached!
                        user = alert.user
                        
                        # Send notification
                        self.send_email_alert(
                            user.email,
                            alert.product_name,
                            alert.target_price,
                            current_price,
                            alert.product_url
                        )
                        
                        # Update alert
                        alert.current_price = current_price
                        alert.is_active = False  # Disable after triggering
                        db.session.commit()
                        
                        print(f"Price alert triggered for user {user.username}")
                    
                    elif current_price:
                        # Update current price
                        alert.current_price = current_price
                        db.session.commit()
                        
                except Exception as e:
                    print(f"Error checking alert {alert.id}: {e}")
                    continue
    
    def get_current_price(self, product_url):
        """Get current price from product URL"""
        try:
            if 'amazon.com' in product_url:
                return self.scraper.get_amazon_price(product_url)
            elif 'ebay.com' in product_url:
                return self.scraper.get_ebay_price(product_url)
            elif 'walmart.com' in product_url:
                return self.scraper.get_walmart_price(product_url)
            else:
                return None
        except Exception as e:
            print(f"Error getting price from {product_url}: {e}")
            return None
    
    def start_price_monitoring(self):
        """Start background thread for price monitoring"""
        def monitor():
            while True:
                try:
                    self.check_price_alerts()
                    time.sleep(3600)  # Check every hour
                except Exception as e:
                    print(f"Error in price monitoring: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def register_routes(self, auth_manager):
        """Register price alert routes"""
        
        @self.app.route('/api/price-alerts', methods=['POST'])
        @auth_manager.require_auth
        def create_price_alert():
            data = request.get_json()
            
            required_fields = ['product_name', 'product_url', 'target_price']
            if not all(field in data for field in required_fields):
                return jsonify({'error': 'Missing required fields'}), 400
            
            alert = PriceAlert(
                user_id=request.current_user_id,
                product_name=data['product_name'],
                product_url=data['product_url'],
                target_price=float(data['target_price']),
                current_price=self.get_current_price(data['product_url'])
            )
            
            db.session.add(alert)
            db.session.commit()
            
            return jsonify({
                'message': 'Price alert created successfully',
                'alert_id': alert.id
            }), 201
        
        @self.app.route('/api/price-alerts', methods=['GET'])
        @auth_manager.require_auth
        def get_price_alerts():
            alerts = PriceAlert.query.filter_by(
                user_id=request.current_user_id
            ).order_by(PriceAlert.created_at.desc()).all()
            
            return jsonify([{
                'id': alert.id,
                'product_name': alert.product_name,
                'product_url': alert.product_url,
                'target_price': alert.target_price,
                'current_price': alert.current_price,
                'is_active': alert.is_active,
                'created_at': alert.created_at.isoformat(),
                'last_checked': alert.last_checked.isoformat() if alert.last_checked else None
            } for alert in alerts])
        
        @self.app.route('/api/price-alerts/<int:alert_id>', methods=['DELETE'])
        @auth_manager.require_auth
        def delete_price_alert(alert_id):
            alert = PriceAlert.query.filter_by(
                id=alert_id,
                user_id=request.current_user_id
            ).first()
            
            if not alert:
                return jsonify({'error': 'Alert not found'}), 404
            
            db.session.delete(alert)
            db.session.commit()
            
            return jsonify({'message': 'Price alert deleted successfully'})
