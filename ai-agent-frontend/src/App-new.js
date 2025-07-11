import React, { useState } from "react";
import './App.css';
import logo from './images/logo.jpg';
import { AuthProvider, useAuth } from './context/AuthContext';
import ProductCard from './components/ProductCard';
import axios from 'axios';

// Configure axios base URL
axios.defaults.baseURL = 'http://127.0.0.1:5000';

// Login Component
function LoginForm() {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    username: ''
  });
  const [error, setError] = useState('');
  const { login, register, loading } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    const result = isLogin 
      ? await login(formData.email, formData.password)
      : await register(formData.username, formData.email, formData.password);
    
    if (!result.success) {
      setError(result.error);
    }
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <img src={logo} alt="AI Shopping Agent" className="auth-logo" />
        <h2>🛒 AI Shopping Agent</h2>
        <p className="auth-subtitle">Your intelligent shopping companion</p>
        
        {error && <div className="error-message">{error}</div>}
        
        <form onSubmit={handleSubmit} className="auth-form">
          {!isLogin && (
            <input
              type="text"
              name="username"
              placeholder="Username"
              value={formData.username}
              onChange={handleChange}
              required
              className="auth-input"
            />
          )}
          
          <input
            type="email"
            name="email"
            placeholder="Email"
            value={formData.email}
            onChange={handleChange}
            required
            className="auth-input"
          />
          
          <input
            type="password"
            name="password"
            placeholder="Password"
            value={formData.password}
            onChange={handleChange}
            required
            className="auth-input"
          />
          
          <button type="submit" disabled={loading} className="auth-button">
            {loading ? 'Please wait...' : (isLogin ? 'Login' : 'Register')}
          </button>
        </form>
        
        <p className="auth-switch">
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <button 
            type="button" 
            onClick={() => setIsLogin(!isLogin)}
            className="link-button"
          >
            {isLogin ? 'Register' : 'Login'}
          </button>
        </p>
      </div>
    </div>
  );
}

// Main Shopping Component
function ShoppingApp() {
  const { user, logout } = useAuth();
  const [criteria, setCriteria] = useState({ budget: "", product: "" });
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sortOption, setSortOption] = useState("price");
  const [filterText, setFilterText] = useState("");
  const [useNLP, setUseNLP] = useState(false);
  const [currency, setCurrency] = useState("USD");
  const [error, setError] = useState(null);

  const getGreeting = () => {
    const hour = new Date().getHours();
    let greeting = '';
    if (hour < 12) greeting = "Good morning";
    else if (hour < 18) greeting = "Good afternoon";
    else greeting = "Good evening";
    return `${greeting}, ${user?.username}`;
  };

  const getAssistantMessage = () => {
    if (!criteria.budget && !criteria.product) {
      return "What are you looking for today?";
    } else if (criteria.budget && !criteria.product) {
      return "Please tell me the product you're looking for.";
    } else if (!criteria.budget && criteria.product) {
      return "Please tell me your budget.";
    }
    return "How may I assist you further?";
  };

  const handleSearch = async () => {
    if (!criteria.budget || !criteria.product) {
      setError("Please enter both budget and product");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('/api/auth/recommend', {
        budget: parseFloat(criteria.budget),
        product: criteria.product,
        use_nlp: useNLP,
        currency: currency,
        language: 'en'
      });

      setResults(response.data);
    } catch (error) {
      setError(error.response?.data?.error || 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleAddToFavorites = async (product) => {
    try {
      await axios.post('/api/favorites', {
        product_name: product.name,
        product_url: product.url,
        price: product.price,
        platform: product.platform
      });
      alert('Added to favorites!');
    } catch (error) {
      console.error('Error adding to favorites:', error);
    }
  };

  const handlePriceAlert = async (product) => {
    const targetPrice = prompt(`Set price alert for "${product.name}"\nCurrent price: $${product.price}\nAlert me when price drops below:`);
    
    if (targetPrice && !isNaN(targetPrice)) {
      try {
        await axios.post('/api/price-alerts', {
          product_name: product.name,
          product_url: product.url,
          target_price: parseFloat(targetPrice)
        });
        alert('Price alert created!');
      } catch (error) {
        console.error('Error creating price alert:', error);
      }
    }
  };

  const filteredResults = results.filter(result =>
    result.name.toLowerCase().includes(filterText.toLowerCase())
  );

  const sortedResults = [...filteredResults].sort((a, b) => {
    if (sortOption === "price") return a.price - b.price;
    if (sortOption === "name") return a.name.localeCompare(b.name);
    if (sortOption === "rating") return (b.rating || 0) - (a.rating || 0);
    return 0;
  });

  return (
    <div className="shopping-app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-section">
            <img src={logo} alt="AI Shopping Agent" className="header-logo" />
            <h1>🛒 AI Shopping Agent</h1>
          </div>
          <div className="user-section">
            <span className="greeting">{getGreeting()}</span>
            <button onClick={logout} className="logout-btn">Logout</button>
          </div>
        </div>
      </header>

      {/* Search Section */}
      <section className="search-section">
        <div className="search-container">
          <h2>Find Your Perfect Product</h2>
          <p className="assistant-message">{getAssistantMessage()}</p>
          
          {error && <div className="error-message">{error}</div>}
          
          <div className="search-form">
            <div className="input-group">
              <input
                type="text"
                placeholder="What are you looking for?"
                value={criteria.product}
                onChange={(e) => setCriteria({...criteria, product: e.target.value})}
                className="search-input"
              />
              
              <input
                type="number"
                placeholder="Budget ($)"
                value={criteria.budget}
                onChange={(e) => setCriteria({...criteria, budget: e.target.value})}
                className="budget-input"
              />
            </div>
            
            <div className="search-options">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={useNLP}
                  onChange={(e) => setUseNLP(e.target.checked)}
                />
                Use Smart Search (NLP)
              </label>
              
              <select 
                value={currency} 
                onChange={(e) => setCurrency(e.target.value)}
                className="currency-select"
              >
                <option value="USD">USD ($)</option>
                <option value="EUR">EUR (€)</option>
                <option value="GBP">GBP (£)</option>
                <option value="LKR">LKR (₨)</option>
              </select>
            </div>
            
            <button 
              onClick={handleSearch} 
              disabled={loading}
              className="search-button"
            >
              {loading ? '🔍 Searching...' : '🔍 Search Products'}
            </button>
          </div>
        </div>
      </section>

      {/* Results Section */}
      {results.length > 0 && (
        <section className="results-section">
          <div className="results-header">
            <h3>Found {results.length} products</h3>
            
            <div className="results-controls">
              <input
                type="text"
                placeholder="Filter results..."
                value={filterText}
                onChange={(e) => setFilterText(e.target.value)}
                className="filter-input"
              />
              
              <select 
                value={sortOption} 
                onChange={(e) => setSortOption(e.target.value)}
                className="sort-select"
              >
                <option value="price">Sort by Price</option>
                <option value="name">Sort by Name</option>
                <option value="rating">Sort by Rating</option>
              </select>
            </div>
          </div>
          
          <div className="products-grid">
            {sortedResults.map((product, index) => (
              <ProductCard
                key={index}
                product={product}
                onAddToFavorites={handleAddToFavorites}
                onPriceAlert={handlePriceAlert}
              />
            ))}
          </div>
        </section>
      )}
      
      <footer className="app-footer">
        <p>© 2025 AI Shopping Agent - Your Smart Shopping Companion</p>
      </footer>
    </div>
  );
}

// Main App Component with Auth Provider
function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

function AppContent() {
  const { isAuthenticated } = useAuth();
  
  return (
    <div className="App">
      {isAuthenticated ? <ShoppingApp /> : <LoginForm />}
    </div>
  );
}

export default App;
