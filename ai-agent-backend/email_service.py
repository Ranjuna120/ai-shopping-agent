"""
Email Notification System for AI Shopping Agent
Handles price alerts, recommendations, and promotional emails
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import sqlite3
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import json
import os
from jinja2 import Template
import schedule
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import smtplib
from email.utils import formataddr
import imaplib
import email
import re
from urllib.parse import quote_plus
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmailTemplate:
    """Email template data structure"""
    subject: str
    html_content: str
    text_content: str
    template_type: str

@dataclass
class EmailNotification:
    """Email notification data structure"""
    recipient_email: str
    subject: str
    html_content: str
    text_content: str
    priority: str = 'normal'  # low, normal, high
    scheduled_time: Optional[datetime] = None
    attachments: List[Dict[str, Any]] = None

class EmailTemplateManager:
    """
    Manages email templates for different notification types
    """
    
    def __init__(self):
        self.templates = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default email templates"""
        
        # Price Alert Template
        self.templates['price_alert'] = EmailTemplate(
            subject="🚨 Price Alert: {{product_name}} is now {{new_price}}!",
            html_content="""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Price Alert</title>
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; }
                    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                    .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; border-radius: 10px 10px 0 0; }
                    .content { background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }
                    .alert-box { background: #4CAF50; color: white; padding: 15px; border-radius: 5px; margin: 20px 0; text-align: center; }
                    .product-info { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    .price-comparison { display: flex; justify-content: space-between; align-items: center; margin: 15px 0; }
                    .old-price { text-decoration: line-through; color: #888; font-size: 18px; }
                    .new-price { color: #4CAF50; font-size: 24px; font-weight: bold; }
                    .savings { background: #FFD700; color: #333; padding: 5px 10px; border-radius: 15px; font-weight: bold; }
                    .btn { display: inline-block; background: #667eea; color: white; padding: 12px 25px; text-decoration: none; border-radius: 5px; margin: 15px 0; }
                    .footer { text-align: center; color: #666; font-size: 12px; margin-top: 30px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🎉 Price Alert Triggered!</h1>
                        <p>Great news! The price has dropped on an item you're watching.</p>
                    </div>
                    <div class="content">
                        <div class="alert-box">
                            <h2>Your target price has been reached!</h2>
                        </div>
                        
                        <div class="product-info">
                            <h3>{{product_name}}</h3>
                            <p><strong>Platform:</strong> {{platform}}</p>
                            
                            <div class="price-comparison">
                                <div>
                                    <div class="old-price">${{old_price}}</div>
                                    <div class="new-price">${{new_price}}</div>
                                </div>
                                <div class="savings">
                                    Save ${{savings}}!
                                </div>
                            </div>
                            
                            <p><strong>Discount:</strong> {{discount_percentage}}% off</p>
                            <p><strong>Alert created:</strong> {{alert_date}}</p>
                        </div>
                        
                        <div style="text-align: center;">
                            <a href="{{product_url}}" class="btn">🛒 Buy Now</a>
                            <a href="{{unsubscribe_url}}" style="color: #666; font-size: 12px; margin-left: 20px;">Unsubscribe from this alert</a>
                        </div>
                        
                        <div class="footer">
                            <p>This alert was set up on {{alert_date}} for {{product_name}}</p>
                            <p>AI Shopping Agent - Your Smart Shopping Companion</p>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """,
            text_content="""
            🚨 PRICE ALERT: {{product_name}}
            
            Great news! The price has dropped on an item you're watching.
            
            Product: {{product_name}}
            Platform: {{platform}}
            
            Old Price: ${{old_price}}
            New Price: ${{new_price}}
            You Save: ${{savings}} ({{discount_percentage}}% off)
            
            Buy Now: {{product_url}}
            
            Alert created: {{alert_date}}
            
            Unsubscribe: {{unsubscribe_url}}
            
            ---
            AI Shopping Agent - Your Smart Shopping Companion
            """,
            template_type='price_alert'
        )
        
        # Recommendation Template
        self.templates['recommendations'] = EmailTemplate(
            subject="🛍️ Personalized Recommendations Just for You!",
            html_content="""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Personalized Recommendations</title>
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; }
                    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                    .header { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; text-align: center; border-radius: 10px 10px 0 0; }
                    .content { background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }
                    .recommendation { background: white; padding: 20px; border-radius: 8px; margin: 15px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    .rec-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
                    .confidence { background: #4CAF50; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; }
                    .reasons { background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; }
                    .price { color: #f5576c; font-size: 20px; font-weight: bold; }
                    .category { color: #666; font-size: 14px; text-transform: uppercase; }
                    .btn { display: inline-block; background: #f5576c; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 10px 5px 0 0; }
                    .footer { text-align: center; color: #666; font-size: 12px; margin-top: 30px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🛍️ Recommended Just for You</h1>
                        <p>Based on your shopping preferences and behavior</p>
                    </div>
                    <div class="content">
                        <p>Hi {{user_name}},</p>
                        <p>We've found some great products that match your interests!</p>
                        
                        {% for rec in recommendations %}
                        <div class="recommendation">
                            <div class="rec-header">
                                <h3>{{rec.product_name}}</h3>
                                <div class="confidence">{{rec.confidence_text}}</div>
                            </div>
                            <div class="category">{{rec.category}}</div>
                            <div class="price">${{rec.estimated_price}}</div>
                            
                            <div class="reasons">
                                <strong>Why we recommend this:</strong>
                                <ul>
                                {% for reason in rec.reasons %}
                                    <li>{{reason}}</li>
                                {% endfor %}
                                </ul>
                            </div>
                            
                            <a href="{{search_url}}&q={{rec.product_name}}" class="btn">🔍 Search</a>
                            <a href="{{add_to_favorites_url}}&product={{rec.product_name}}" class="btn">❤️ Add to Favorites</a>
                        </div>
                        {% endfor %}
                        
                        <div class="footer">
                            <p>These recommendations are generated based on your shopping history and preferences.</p>
                            <p><a href="{{unsubscribe_url}}">Unsubscribe</a> from recommendation emails</p>
                            <p>AI Shopping Agent - Your Smart Shopping Companion</p>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """,
            text_content="""
            🛍️ PERSONALIZED RECOMMENDATIONS
            
            Hi {{user_name}},
            
            We've found some great products that match your interests!
            
            {% for rec in recommendations %}
            ---
            {{rec.product_name}}
            Category: {{rec.category}}
            Estimated Price: ${{rec.estimated_price}}
            Confidence: {{rec.confidence_text}}
            
            Why we recommend this:
            {% for reason in rec.reasons %}
            - {{reason}}
            {% endfor %}
            
            Search: {{search_url}}&q={{rec.product_name}}
            {% endfor %}
            
            ---
            Unsubscribe: {{unsubscribe_url}}
            AI Shopping Agent - Your Smart Shopping Companion
            """,
            template_type='recommendations'
        )
        
        # Weekly Summary Template
        self.templates['weekly_summary'] = EmailTemplate(
            subject="📊 Your Weekly Shopping Summary",
            html_content="""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Weekly Summary</title>
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; }
                    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                    .header { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 20px; text-align: center; border-radius: 10px 10px 0 0; }
                    .content { background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }
                    .stat-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin: 20px 0; }
                    .stat-box { background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    .stat-number { font-size: 24px; font-weight: bold; color: #4facfe; }
                    .stat-label { color: #666; font-size: 14px; }
                    .section { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; }
                    .footer { text-align: center; color: #666; font-size: 12px; margin-top: 30px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>📊 Your Weekly Shopping Summary</h1>
                        <p>{{week_start}} - {{week_end}}</p>
                    </div>
                    <div class="content">
                        <div class="stat-grid">
                            <div class="stat-box">
                                <div class="stat-number">{{total_searches}}</div>
                                <div class="stat-label">Searches</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-number">{{new_favorites}}</div>
                                <div class="stat-label">New Favorites</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-number">{{active_alerts}}</div>
                                <div class="stat-label">Price Alerts</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-number">${{total_savings}}</div>
                                <div class="stat-label">Potential Savings</div>
                            </div>
                        </div>
                        
                        <div class="section">
                            <h3>🔥 Top Categories This Week</h3>
                            {% for category, count in top_categories %}
                            <p>{{category}}: {{count}} searches</p>
                            {% endfor %}
                        </div>
                        
                        <div class="section">
                            <h3>💰 Best Deals Found</h3>
                            {% for deal in best_deals %}
                            <p><strong>{{deal.product_name}}</strong> - ${{deal.price}} ({{deal.discount}}% off)</p>
                            {% endfor %}
                        </div>
                        
                        <div class="footer">
                            <p>Keep shopping smart with AI Shopping Agent!</p>
                            <p><a href="{{unsubscribe_url}}">Unsubscribe</a> from weekly summaries</p>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """,
            text_content="""
            📊 YOUR WEEKLY SHOPPING SUMMARY
            {{week_start}} - {{week_end}}
            
            This Week's Activity:
            - Searches: {{total_searches}}
            - New Favorites: {{new_favorites}}
            - Price Alerts: {{active_alerts}}
            - Potential Savings: ${{total_savings}}
            
            🔥 Top Categories:
            {% for category, count in top_categories %}
            {{category}}: {{count}} searches
            {% endfor %}
            
            💰 Best Deals Found:
            {% for deal in best_deals %}
            {{deal.product_name}} - ${{deal.price}} ({{deal.discount}}% off)
            {% endfor %}
            
            Keep shopping smart with AI Shopping Agent!
            Unsubscribe: {{unsubscribe_url}}
            """,
            template_type='weekly_summary'
        )
    
    def get_template(self, template_type: str) -> Optional[EmailTemplate]:
        """Get email template by type"""
        return self.templates.get(template_type)
    
    def render_template(self, template_type: str, context: Dict[str, Any]) -> Optional[EmailNotification]:
        """Render email template with context data"""
        template = self.get_template(template_type)
        if not template:
            return None
        
        try:
            # Use Jinja2 for template rendering
            subject_template = Template(template.subject)
            html_template = Template(template.html_content)
            text_template = Template(template.text_content)
            
            rendered_subject = subject_template.render(**context)
            rendered_html = html_template.render(**context)
            rendered_text = text_template.render(**context)
            
            return EmailNotification(
                recipient_email=context.get('recipient_email', ''),
                subject=rendered_subject,
                html_content=rendered_html,
                text_content=rendered_text,
                priority=context.get('priority', 'normal')
            )
            
        except Exception as e:
            logger.error(f"Error rendering template {template_type}: {e}")
            return None

class EmailSender:
    """
    Handles sending emails through SMTP
    """
    
    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_server = smtp_config.get('server', 'smtp.gmail.com')
        self.smtp_port = smtp_config.get('port', 587)
        self.username = smtp_config.get('username', '')
        self.password = smtp_config.get('password', '')
        self.use_tls = smtp_config.get('use_tls', True)
        self.sender_name = smtp_config.get('sender_name', 'AI Shopping Agent')
        
        # Email queue for batch processing
        self.email_queue = queue.Queue()
        self.max_workers = 5
        
    def send_email(self, notification: EmailNotification) -> bool:
        """Send a single email"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = notification.subject
            msg['From'] = formataddr((self.sender_name, self.username))
            msg['To'] = notification.recipient_email
            
            # Add text and HTML parts
            text_part = MIMEText(notification.text_content, 'plain')
            html_part = MIMEText(notification.html_content, 'html')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Add attachments if any
            if notification.attachments:
                for attachment in notification.attachments:
                    self._add_attachment(msg, attachment)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {notification.recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {notification.recipient_email}: {e}")
            return False
    
    def _add_attachment(self, msg: MIMEMultipart, attachment: Dict[str, Any]):
        """Add attachment to email message"""
        try:
            with open(attachment['filepath'], 'rb') as f:
                part = MimeBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {attachment["filename"]}'
                )
                msg.attach(part)
        except Exception as e:
            logger.error(f"Failed to add attachment {attachment['filename']}: {e}")
    
    def send_bulk_emails(self, notifications: List[EmailNotification]) -> Dict[str, int]:
        """Send multiple emails concurrently"""
        results = {'sent': 0, 'failed': 0}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all email sending tasks
            future_to_email = {
                executor.submit(self.send_email, notification): notification
                for notification in notifications
            }
            
            # Collect results
            for future in as_completed(future_to_email):
                notification = future_to_email[future]
                try:
                    success = future.result()
                    if success:
                        results['sent'] += 1
                    else:
                        results['failed'] += 1
                except Exception as e:
                    logger.error(f"Error sending email to {notification.recipient_email}: {e}")
                    results['failed'] += 1
        
        return results
    
    def queue_email(self, notification: EmailNotification):
        """Add email to queue for batch processing"""
        self.email_queue.put(notification)
    
    def process_queue(self) -> Dict[str, int]:
        """Process all emails in the queue"""
        notifications = []
        
        # Collect all emails from queue
        while not self.email_queue.empty():
            try:
                notification = self.email_queue.get_nowait()
                notifications.append(notification)
            except queue.Empty:
                break
        
        if notifications:
            return self.send_bulk_emails(notifications)
        
        return {'sent': 0, 'failed': 0}

class PriceAlertMonitor:
    """
    Monitors price alerts and sends notifications
    """
    
    def __init__(self, db_path: str, email_sender: EmailSender, template_manager: EmailTemplateManager):
        self.db_path = db_path
        self.email_sender = email_sender
        self.template_manager = template_manager
        
    def get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def check_price_alerts(self):
        """Check all active price alerts and send notifications"""
        conn = self.get_db_connection()
        
        # Get active price alerts
        query = """
        SELECT pa.*, u.email, u.username 
        FROM price_alert pa
        JOIN users u ON pa.user_id = u.id
        WHERE pa.is_active = 1
        AND (pa.last_checked IS NULL OR pa.last_checked < ?)
        """
        
        # Check alerts that haven't been checked in the last hour
        last_check_time = datetime.now() - timedelta(hours=1)
        
        cursor = conn.execute(query, (last_check_time.isoformat(),))
        alerts = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        notifications_sent = 0
        
        for alert_row in alerts:
            alert = dict(zip(columns, alert_row))
            
            # Simulate price checking (in real implementation, this would scrape the actual price)
            current_price = self._get_current_price(alert['product_url'])
            
            if current_price and current_price <= alert['target_price']:
                # Price target reached, send notification
                notification = self._create_price_alert_notification(alert, current_price)
                
                if notification:
                    success = self.email_sender.send_email(notification)
                    if success:
                        notifications_sent += 1
                        # Update alert status
                        self._update_alert_status(conn, alert['id'], current_price)
            else:
                # Update last checked time
                self._update_last_checked(conn, alert['id'])
        
        conn.close()
        logger.info(f"Checked price alerts, sent {notifications_sent} notifications")
        return notifications_sent
    
    def _get_current_price(self, product_url: str) -> Optional[float]:
        """
        Get current price from product URL
        This is a simplified version - real implementation would use web scraping
        """
        # Simulate price fluctuation for demo purposes
        import random
        base_price = random.uniform(50, 500)
        # 30% chance of price drop
        if random.random() < 0.3:
            return base_price * random.uniform(0.7, 0.9)  # 10-30% discount
        return base_price
    
    def _create_price_alert_notification(self, alert: Dict[str, Any], current_price: float) -> Optional[EmailNotification]:
        """Create price alert email notification"""
        old_price = alert.get('current_price', 0)
        savings = old_price - current_price
        discount_percentage = (savings / old_price * 100) if old_price > 0 else 0
        
        context = {
            'recipient_email': alert['email'],
            'product_name': alert.get('product_name', 'Product'),
            'platform': self._extract_platform_from_url(alert.get('product_url', '')),
            'old_price': f"{old_price:.2f}",
            'new_price': f"{current_price:.2f}",
            'savings': f"{savings:.2f}",
            'discount_percentage': f"{discount_percentage:.1f}",
            'alert_date': alert.get('created_at', ''),
            'product_url': alert.get('product_url', '#'),
            'unsubscribe_url': f"http://localhost:5000/unsubscribe?alert_id={alert['id']}"
        }
        
        return self.template_manager.render_template('price_alert', context)
    
    def _extract_platform_from_url(self, url: str) -> str:
        """Extract platform name from URL"""
        if 'amazon' in url.lower():
            return 'Amazon'
        elif 'ebay' in url.lower():
            return 'eBay'
        elif 'walmart' in url.lower():
            return 'Walmart'
        else:
            return 'Unknown Platform'
    
    def _update_alert_status(self, conn: sqlite3.Connection, alert_id: int, current_price: float):
        """Update alert status after notification sent"""
        query = """
        UPDATE price_alert 
        SET is_active = 0, last_checked = ?, current_price = ?, triggered_at = ?
        WHERE id = ?
        """
        
        now = datetime.now().isoformat()
        conn.execute(query, (now, current_price, now, alert_id))
        conn.commit()
    
    def _update_last_checked(self, conn: sqlite3.Connection, alert_id: int):
        """Update last checked time for alert"""
        query = "UPDATE price_alert SET last_checked = ? WHERE id = ?"
        conn.execute(query, (datetime.now().isoformat(), alert_id))
        conn.commit()

class RecommendationEmailer:
    """
    Sends personalized recommendation emails
    """
    
    def __init__(self, db_path: str, email_sender: EmailSender, template_manager: EmailTemplateManager):
        self.db_path = db_path
        self.email_sender = email_sender
        self.template_manager = template_manager
    
    def send_weekly_recommendations(self):
        """Send weekly recommendation emails to active users"""
        from ml_engine import RecommendationEngine
        
        # Get ML recommendations
        rec_engine = RecommendationEngine(self.db_path)
        rec_engine.load_models()
        
        # Get active users
        active_users = self._get_active_users()
        
        notifications_sent = 0
        
        for user in active_users:
            recommendations = rec_engine.get_recommendations(user['id'], 5)
            
            if recommendations:
                notification = self._create_recommendation_notification(user, recommendations)
                
                if notification:
                    success = self.email_sender.send_email(notification)
                    if success:
                        notifications_sent += 1
        
        logger.info(f"Sent {notifications_sent} recommendation emails")
        return notifications_sent
    
    def _get_active_users(self) -> List[Dict[str, Any]]:
        """Get list of active users who want recommendation emails"""
        conn = sqlite3.connect(self.db_path)
        
        # Get users who have been active in the last 30 days
        query = """
        SELECT DISTINCT u.id, u.email, u.username
        FROM users u
        WHERE u.email IS NOT NULL
        AND u.email != ''
        AND EXISTS (
            SELECT 1 FROM search_history sh 
            WHERE sh.user_id = u.id 
            AND sh.created_at > date('now', '-30 days')
        )
        """
        
        cursor = conn.execute(query)
        columns = [description[0] for description in cursor.description]
        
        users = []
        for row in cursor.fetchall():
            users.append(dict(zip(columns, row)))
        
        conn.close()
        return users
    
    def _create_recommendation_notification(self, user: Dict[str, Any], recommendations) -> Optional[EmailNotification]:
        """Create recommendation email notification"""
        # Prepare recommendation data for template
        rec_data = []
        for rec in recommendations:
            confidence_text = f"{rec.confidence * 100:.0f}% match"
            rec_data.append({
                'product_name': rec.product_name,
                'category': rec.category,
                'estimated_price': f"{rec.estimated_price:.2f}",
                'confidence_text': confidence_text,
                'reasons': rec.reasons[:3]  # Limit to top 3 reasons
            })
        
        context = {
            'recipient_email': user['email'],
            'user_name': user.get('username', 'Valued Customer'),
            'recommendations': rec_data,
            'search_url': 'http://localhost:3000/search',
            'add_to_favorites_url': 'http://localhost:3000/favorites/add',
            'unsubscribe_url': f"http://localhost:5000/unsubscribe?user_id={user['id']}&type=recommendations"
        }
        
        return self.template_manager.render_template('recommendations', context)

class WeeklySummaryGenerator:
    """
    Generates and sends weekly summary emails
    """
    
    def __init__(self, db_path: str, email_sender: EmailSender, template_manager: EmailTemplateManager):
        self.db_path = db_path
        self.email_sender = email_sender
        self.template_manager = template_manager
    
    def send_weekly_summaries(self):
        """Send weekly summary emails to all users"""
        active_users = self._get_active_users()
        
        notifications_sent = 0
        
        for user in active_users:
            summary_data = self._generate_user_summary(user['id'])
            
            if summary_data:
                notification = self._create_summary_notification(user, summary_data)
                
                if notification:
                    success = self.email_sender.send_email(notification)
                    if success:
                        notifications_sent += 1
        
        logger.info(f"Sent {notifications_sent} weekly summary emails")
        return notifications_sent
    
    def _get_active_users(self) -> List[Dict[str, Any]]:
        """Get list of active users"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT id, email, username
        FROM users
        WHERE email IS NOT NULL AND email != ''
        """
        
        cursor = conn.execute(query)
        columns = [description[0] for description in cursor.description]
        
        users = []
        for row in cursor.fetchall():
            users.append(dict(zip(columns, row)))
        
        conn.close()
        return users
    
    def _generate_user_summary(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Generate weekly summary data for user"""
        conn = sqlite3.connect(self.db_path)
        
        # Date range for this week
        today = datetime.now()
        week_start = today - timedelta(days=7)
        
        # Get search statistics
        search_query = """
        SELECT COUNT(*) as total_searches
        FROM search_history
        WHERE user_id = ? AND created_at > ?
        """
        total_searches = conn.execute(search_query, (user_id, week_start.isoformat())).fetchone()[0]
        
        # Get favorites statistics
        favorites_query = """
        SELECT COUNT(*) as new_favorites
        FROM favorites
        WHERE user_id = ? AND added_at > ?
        """
        new_favorites = conn.execute(favorites_query, (user_id, week_start.isoformat())).fetchone()[0]
        
        # Get active alerts
        alerts_query = """
        SELECT COUNT(*) as active_alerts
        FROM price_alert
        WHERE user_id = ? AND is_active = 1
        """
        active_alerts = conn.execute(alerts_query, (user_id,)).fetchone()[0]
        
        # Get top categories
        category_query = """
        SELECT query, COUNT(*) as count
        FROM search_history
        WHERE user_id = ? AND created_at > ?
        GROUP BY query
        ORDER BY count DESC
        LIMIT 5
        """
        categories = conn.execute(category_query, (user_id, week_start.isoformat())).fetchall()
        
        conn.close()
        
        if total_searches == 0 and new_favorites == 0:
            return None  # User was not active this week
        
        summary = {
            'week_start': week_start.strftime('%B %d'),
            'week_end': today.strftime('%B %d, %Y'),
            'total_searches': total_searches,
            'new_favorites': new_favorites,
            'active_alerts': active_alerts,
            'total_savings': '50.00',  # Placeholder - would calculate from actual data
            'top_categories': [(cat[0][:20], cat[1]) for cat in categories[:3]],
            'best_deals': [
                {'product_name': 'Sample Product', 'price': '99.99', 'discount': '25'}
            ]  # Placeholder - would get from actual deals
        }
        
        return summary
    
    def _create_summary_notification(self, user: Dict[str, Any], summary_data: Dict[str, Any]) -> Optional[EmailNotification]:
        """Create weekly summary email notification"""
        context = {
            'recipient_email': user['email'],
            'unsubscribe_url': f"http://localhost:5000/unsubscribe?user_id={user['id']}&type=weekly_summary",
            **summary_data
        }
        
        return self.template_manager.render_template('weekly_summary', context)

class EmailNotificationService:
    """
    Main email notification service that coordinates all email functionality
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.db_path = config.get('db_path', 'shopping_agent.db')
        
        # Initialize components
        smtp_config = config.get('smtp', {})
        self.email_sender = EmailSender(smtp_config)
        self.template_manager = EmailTemplateManager()
        
        # Initialize monitors and generators
        self.price_alert_monitor = PriceAlertMonitor(
            self.db_path, self.email_sender, self.template_manager
        )
        self.recommendation_emailer = RecommendationEmailer(
            self.db_path, self.email_sender, self.template_manager
        )
        self.summary_generator = WeeklySummaryGenerator(
            self.db_path, self.email_sender, self.template_manager
        )
        
        # Scheduler for automated tasks
        self.scheduler_thread = None
        self.running = False
    
    def start_scheduler(self):
        """Start automated email scheduler"""
        if self.running:
            return
        
        self.running = True
        
        # Schedule tasks
        schedule.every(1).hours.do(self.price_alert_monitor.check_price_alerts)
        schedule.every().monday.at("09:00").do(self.recommendation_emailer.send_weekly_recommendations)
        schedule.every().sunday.at("18:00").do(self.summary_generator.send_weekly_summaries)
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info("Email notification scheduler started")
    
    def stop_scheduler(self):
        """Stop automated email scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        logger.info("Email notification scheduler stopped")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def send_immediate_price_alert(self, alert_id: int) -> bool:
        """Send immediate price alert notification"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT pa.*, u.email, u.username 
        FROM price_alert pa
        JOIN users u ON pa.user_id = u.id
        WHERE pa.id = ?
        """
        
        result = conn.execute(query, (alert_id,)).fetchone()
        conn.close()
        
        if not result:
            return False
        
        columns = ['id', 'user_id', 'product_url', 'target_price', 'current_price', 
                  'product_name', 'is_active', 'created_at', 'last_checked', 'email', 'username']
        alert = dict(zip(columns, result))
        
        # Get current price (simulate)
        current_price = alert['target_price'] - 10  # Simulate price drop
        
        notification = self.price_alert_monitor._create_price_alert_notification(alert, current_price)
        
        if notification:
            return self.email_sender.send_email(notification)
        
        return False

# Flask route integration
def create_email_routes(app, db):
    """Create email notification routes for Flask app"""
    
    # Load email configuration
    email_config = {
        'db_path': 'shopping_agent.db',
        'smtp': {
            'server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'port': int(os.getenv('SMTP_PORT', '587')),
            'username': os.getenv('SMTP_USERNAME', ''),
            'password': os.getenv('SMTP_PASSWORD', ''),
            'use_tls': os.getenv('SMTP_USE_TLS', 'true').lower() == 'true',
            'sender_name': os.getenv('SENDER_NAME', 'AI Shopping Agent')
        }
    }
    
    # Initialize email service
    email_service = EmailNotificationService(email_config)
    email_service.start_scheduler()
    
    @app.route('/api/email/test-alert/<int:alert_id>', methods=['POST'])
    def test_price_alert(alert_id):
        """Test price alert email"""
        try:
            success = email_service.send_immediate_price_alert(alert_id)
            return {
                'success': success,
                'message': 'Test email sent' if success else 'Failed to send email'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500
    
    @app.route('/api/email/send-recommendations/<int:user_id>', methods=['POST'])
    def send_user_recommendations(user_id):
        """Send recommendations to specific user"""
        try:
            # Get user email
            conn = sqlite3.connect(email_service.db_path)
            result = conn.execute("SELECT email, username FROM users WHERE id = ?", (user_id,)).fetchone()
            conn.close()
            
            if not result:
                return {'success': False, 'error': 'User not found'}, 404
            
            user = {'id': user_id, 'email': result[0], 'username': result[1]}
            
            # Send recommendations
            from ml_engine import RecommendationEngine
            rec_engine = RecommendationEngine(email_service.db_path)
            recommendations = rec_engine.get_recommendations(user_id, 5)
            
            if recommendations:
                notification = email_service.recommendation_emailer._create_recommendation_notification(user, recommendations)
                if notification:
                    success = email_service.email_sender.send_email(notification)
                    return {'success': success}
            
            return {'success': False, 'error': 'No recommendations available'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500
    
    @app.route('/unsubscribe')
    def unsubscribe():
        """Handle email unsubscribe requests"""
        from flask import request
        alert_id = request.args.get('alert_id')
        user_id = request.args.get('user_id')
        email_type = request.args.get('type', 'all')
        
        try:
            conn = sqlite3.connect(email_service.db_path)
            
            if alert_id:
                # Deactivate specific price alert
                conn.execute("UPDATE price_alert SET is_active = 0 WHERE id = ?", (alert_id,))
                message = "Price alert deactivated successfully"
            elif user_id:
                # Handle user preferences (would need a preferences table)
                message = f"Unsubscribed from {email_type} emails"
            else:
                message = "Invalid unsubscribe request"
            
            conn.commit()
            conn.close()
            
            return f"""
            <html>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                <h2>Unsubscribed Successfully</h2>
                <p>{message}</p>
                <p><a href="http://localhost:3000">Return to AI Shopping Agent</a></p>
            </body>
            </html>
            """
            
        except Exception as e:
            return f"Error: {str(e)}", 500

if __name__ == "__main__":
    # Test the email system
    config = {
        'db_path': 'shopping_agent.db',
        'smtp': {
            'server': 'smtp.gmail.com',
            'port': 587,
            'username': 'your-email@gmail.com',
            'password': 'your-app-password',
            'use_tls': True,
            'sender_name': 'AI Shopping Agent'
        }
    }
    
    email_service = EmailNotificationService(config)
    print("Email notification system initialized")
    print("Available templates:", list(email_service.template_manager.templates.keys()))
