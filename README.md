# 🛒 AI Shopping Agent

An intelligent shopping assistant that helps users find the best products across multiple e-commerce platforms using AI, web scraping, and voice recognition.

## ✨ Features

### 🔐 Authentication & User Management
- 🔒 **Secure Authentication**: JWT-based user registration and login system
- 👤 **User Profiles**: Personal user accounts with secure session management
- 🛡️ **Protected Routes**: Secure API endpoints with authentication middleware

### 🛍️ Advanced Shopping Features
- 🤖 **AI-Powered Recommendations**: Uses OpenAI GPT for intelligent product suggestions
- 🛍️ **Multi-Platform Scraping**: Enhanced scraping across Amazon, eBay, and Walmart
- 💝 **Favorites Management**: Save and organize your favorite products
- 🚨 **Price Alerts**: Set target prices and get email notifications when prices drop
- 🔍 **Smart Search**: Real-time product search with enhanced results
- 📊 **Price Comparison**: Dynamic price comparison across multiple platforms

### 🎨 Modern User Experience
- 🎨 **Beautiful UI**: Modern React 18 frontend with glass morphism design
- � **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- ⚡ **Real-time Updates**: Live product data and instant search results
- 🎭 **Interactive Components**: Smooth animations and user-friendly interface

### 🔧 Technical Features
- �🗣️ **Voice Commands**: Voice-enabled product search using speech recognition
- 💱 **Currency Conversion**: Real-time currency conversion for international shopping
- 🔍 **NLP Processing**: Natural language processing for better search queries
- � **Email Notifications**: Automated email alerts for price drops
- 💾 **Data Persistence**: SQLite database with comprehensive data models

## 🏗️ Project Structure

```
ai-shopping-agent/
├── ai-agent-frontend/          # React 18 frontend application
│   ├── src/
│   │   ├── App.js             # Main application component
│   │   ├── Login.js           # Authentication component
│   │   ├── App.css            # Modern styling with glass morphism
│   │   └── images/            # Application assets
│   ├── public/
│   ├── package.json
│   └── README.md
├── ai-agent-backend/           # Flask backend API
│   ├── app.py                 # Main application server
│   ├── models.py              # Database models (User, Favorites, PriceAlert)
│   ├── auth.py                # JWT authentication manager
│   ├── enhanced_scraper.py    # Multi-platform web scraper
│   ├── price_alerts.py        # Price monitoring system
│   ├── requirements.txt       # Python dependencies
│   └── .env.example          # Environment variables template
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- **Node.js** (v16 or higher)
- **Python** (v3.8 or higher)
- **Git**

### 🔧 Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd ai-agent-backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env file and add your configuration:
   # - OpenAI API key
   # - Email credentials for price alerts
   # - Database URL (optional, defaults to SQLite)
   ```

4. Initialize the database:
   ```bash
   # The database will be automatically created on first run
   python app.py
   ```

5. Start the Flask server:
   ```bash
   python app.py
   ```
   The backend will run on `http://127.0.0.1:5000` with the following message:
   ```
   🚀 Starting AI Shopping Agent Backend...
   📡 Available at: http://127.0.0.1:5000
   🔗 Test endpoint: http://127.0.0.1:5000/api/test
   ```

### 🌐 Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd ai-agent-frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the React development server:
   ```bash
   npm start
   ```
   The frontend will run on `http://localhost:3000`

## 🎯 How to Use

### 1. Getting Started
1. **Register/Login**: Create an account or sign in to access features
2. **Search Products**: Use the search bar to find products across multiple platforms
3. **Browse Results**: View products from Amazon, eBay, and Walmart with prices and images

### 2. Managing Favorites
1. **Add to Favorites**: Click the heart icon on any product to save it
2. **View Favorites**: Navigate to the "My Favorites" section
3. **Remove Favorites**: Click the remove button to delete saved products
4. **Organize**: Keep track of products you're interested in purchasing

### 3. Setting Price Alerts
1. **Create Alert**: Click "Set Price Alert" on any product
2. **Set Target Price**: Enter your desired price point
3. **Get Notifications**: Receive email alerts when prices drop below your target
4. **Manage Alerts**: View, edit, or delete your active price alerts
5. **Monitor Prices**: Background system continuously checks for price changes

### 4. Advanced Features
1. **Voice Search**: Use voice commands to search for products
2. **Currency Conversion**: View prices in different currencies
3. **AI Recommendations**: Get intelligent product suggestions based on budget and preferences
4. **Real-time Updates**: See live price data and instant search results

## 📡 API Endpoints

### Authentication
```http
# Register a new user
POST /api/auth/register
Content-Type: application/json

{
  "username": "your_username",
  "email": "your_email@example.com",
  "password": "your_password"
}

# Login user
POST /api/auth/login
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

### Product Search & Management
```http
# Search products across platforms
GET /api/products/search?query=laptop&max_results=10
Authorization: Bearer <jwt_token>

# Add product to favorites
POST /api/favorites
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "name": "Product Name",
  "url": "https://example.com/product",
  "price": 299.99,
  "platform": "Amazon"
}

# Get user favorites
GET /api/favorites
Authorization: Bearer <jwt_token>

# Remove from favorites
DELETE /api/favorites/<favorite_id>
Authorization: Bearer <jwt_token>
```

### Price Alerts
```http
# Create price alert
POST /api/price-alerts
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "product_name": "Product Name",
  "product_url": "https://example.com/product",
  "target_price": 250.00
}

# Get user price alerts
GET /api/price-alerts
Authorization: Bearer <jwt_token>

# Update price alert
PUT /api/price-alerts/<alert_id>
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "target_price": 200.00,
  "is_active": true
}

# Delete price alert
DELETE /api/price-alerts/<alert_id>
Authorization: Bearer <jwt_token>
```

### Legacy Endpoints (AI Recommendations)
```http
# AI-powered product recommendations
POST /api/auth/recommend
Content-Type: application/json

{
  "budget": 500,
  "product": "laptop",
  "use_nlp": true,
  "currency": "USD",
  "language": "en"
}

# Voice search functionality
GET /api/auth/voice_search

# Fetch and compare products
GET /api/auth/fetch_and_compare
```

## 🛠️ Technologies Used

### Frontend
- **React 18** - Modern UI framework with hooks and context
- **Axios** - HTTP client for API communication
- **CSS3** - Modern styling with glass morphism effects
- **JavaScript ES6+** - Modern JavaScript features

### Backend
- **Flask 2.3+** - Lightweight web framework
- **SQLAlchemy 3.0+** - Database ORM with modern syntax
- **SQLite** - Embedded database for data persistence
- **JWT** - JSON Web Tokens for secure authentication
- **Flask-CORS** - Cross-origin resource sharing

### Web Scraping & AI
- **Selenium 4.15+** - Advanced web automation and scraping
- **Chrome WebDriver** - Headless browser automation
- **OpenAI GPT** - AI-powered product recommendations
- **BeautifulSoup** - HTML parsing and data extraction
- **spaCy** - Natural language processing
- **Requests** - HTTP library for API calls

### Additional Features
- **Email Notifications** - SMTP email alerts for price drops
- **Background Threading** - Concurrent price monitoring
- **SpeechRecognition** - Voice command processing
- **forex-python** - Real-time currency conversion
- **ThreadPoolExecutor** - Concurrent web scraping

## 🔒 Security Notes

- Never commit API keys to version control
- The OpenAI API key in the code should be replaced with your own
- Consider using environment variables for sensitive data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Troubleshooting

### Common Issues

1. **Authentication errors**: 
   - Make sure you're registered and logged in
   - Check that JWT tokens are being sent with requests
   - Verify the backend is running on port 5000

2. **Database errors**: 
   - Database is automatically created on first run
   - If you see "no such table" errors, restart the backend
   - Delete the database file and restart to reset schema

3. **Price alerts not working**:
   - Verify email credentials are set in `.env` file
   - Check that the target product URL is accessible
   - Ensure background monitoring thread is running

4. **Scraping failures**:
   - Chrome WebDriver is automatically managed
   - Some sites may block automated requests
   - Try different product URLs if scraping fails

5. **CORS errors**: 
   - Ensure Flask-CORS is properly configured
   - Backend should allow requests from frontend port

6. **API key errors**: 
   - Verify your OpenAI API key is set correctly in `.env`
   - Check API key permissions and usage limits

7. **Module not found errors**: 
   - Run `pip install -r requirements.txt` in backend
   - Run `npm install` in frontend directory

8. **Port conflicts**:
   - Backend runs on port 5000
   - Frontend runs on port 3000 (or 3001 if 3000 is busy)
   - Use different ports if these are occupied

### Database Schema

The application uses the following main tables:
- **users**: User accounts and authentication
- **favorites**: User's saved favorite products  
- **price_alert**: Price monitoring alerts
- **search_history**: User's search history
- **user_preference**: User preferences and settings

### Getting Help

If you encounter any issues:
1. Check the browser console for frontend errors
2. Check the Flask terminal for backend errors
3. Ensure both frontend and backend servers are running
4. Verify all dependencies are installed correctly
5. Check the Simple Browser for live application testing

---

**Created with ❤️ for smart shopping experiences**
