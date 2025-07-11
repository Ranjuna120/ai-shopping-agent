# 🛒 AI Shopping Agent

An intelligent shopping assistant that helps users find the best products across multiple e-commerce platforms using AI, web scraping, and voice recognition.

## ✨ Features

- 🤖 **AI-Powered Recommendations**: Uses OpenAI GPT for intelligent product suggestions
- 🛍️ **Multi-Platform Scraping**: Searches across eBay, Walmart, and other platforms
- 🗣️ **Voice Commands**: Voice-enabled product search using speech recognition
- 💱 **Currency Conversion**: Real-time currency conversion for international shopping
- 🔍 **NLP Processing**: Natural language processing for better search queries
- 📊 **Price Comparison**: Dynamic price comparison across platforms
- 🎨 **Modern UI**: Beautiful React frontend with responsive design

## 🏗️ Project Structure

```
ai-shopping-agent/
├── ai-agent-frontend/          # React frontend application
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── README.md
├── ai-agent-backend/           # Flask backend API
│   ├── app.py
│   └── requirements.txt
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

3. Set up your OpenAI API key:
   - Edit `app.py` and replace `'your-openai-api-key'` with your actual OpenAI API key
   - Or create a `.env` file with your API key

4. Start the Flask server:
   ```bash
   python app.py
   ```
   The backend will run on `http://127.0.0.1:5000`

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

## 📡 API Endpoints

### Product Recommendations
```http
POST /api/auth/recommend
Content-Type: application/json

{
  "budget": 500,
  "product": "laptop",
  "use_nlp": true,
  "currency": "USD",
  "language": "en"
}
```

### Voice Search
```http
GET /api/auth/voice_search
```

### Fetch and Compare Products
```http
GET /api/auth/fetch_and_compare
```

## 🛠️ Technologies Used

### Frontend
- **React** - UI framework
- **Axios** - HTTP client
- **CSS3** - Styling

### Backend
- **Flask** - Web framework
- **OpenAI** - AI recommendations
- **Selenium** - Web scraping
- **BeautifulSoup** - HTML parsing
- **spaCy** - Natural language processing
- **SpeechRecognition** - Voice commands
- **forex-python** - Currency conversion

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

1. **Module not found errors**: Make sure all dependencies are installed
2. **CORS errors**: Ensure Flask-CORS is properly configured
3. **API key errors**: Verify your OpenAI API key is set correctly
4. **Selenium errors**: Chrome WebDriver will be automatically downloaded

### Getting Help

If you encounter any issues:
1. Check the console logs for errors
2. Ensure both frontend and backend servers are running
3. Verify all dependencies are installed correctly

---

**Created with ❤️ for smart shopping experiences**
