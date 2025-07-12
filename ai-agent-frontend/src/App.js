import React, { useState, useEffect } from "react";
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
  const [showFavorites, setShowFavorites] = useState(false);
  const [favorites, setFavorites] = useState([]);
  const [favoritesLoading, setFavoritesLoading] = useState(false);
  const [showPriceAlerts, setShowPriceAlerts] = useState(false);
  const [priceAlerts, setPriceAlerts] = useState([]);
  const [alertsLoading, setAlertsLoading] = useState(false);

  // Load favorites and price alerts when component mounts or user changes
  useEffect(() => {
    if (user) {
      loadFavorites();
      loadPriceAlerts();
    }
  }, [user]);

  // Auto-refresh favorites and price alerts every 30 seconds
  useEffect(() => {
    if (user) {
      const interval = setInterval(() => {
        loadFavorites();
        loadPriceAlerts();
      }, 30000); // 30 seconds

      return () => clearInterval(interval);
    }
  }, [user]);

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
      const token = localStorage.getItem('token');
      await axios.post('/api/favorites', {
        product_name: product.name,
        product_url: product.url,
        price: product.price,
        platform: product.platform
      }, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      alert('Added to favorites!');
      // Reload favorites to update count
      loadFavorites();
    } catch (error) {
      console.error('Error adding to favorites:', error);
      if (error.response?.status === 409) {
        alert('Product already in favorites!');
      } else if (error.response?.status === 401) {
        alert('Please login again to add favorites');
        logout();
      } else {
        alert('Failed to add to favorites. Please try again.');
      }
    }
  };

  const handlePriceAlert = async (product) => {
    const targetPrice = prompt(`🔔 Set Price Alert for "${product.name}"\n\nCurrent Price: $${product.price}\nAlert me when price drops below: $`);
    
    if (targetPrice && !isNaN(targetPrice)) {
      const target = parseFloat(targetPrice);
      
      if (target >= product.price) {
        alert('⚠️ Target price must be lower than current price!');
        return;
      }
      
      try {
        const token = localStorage.getItem('token');
        await axios.post('/api/price-alerts', {
          product_name: product.name,
          product_url: product.url,
          target_price: target
        }, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        alert(`✅ Price alert set! You'll be notified when price drops below $${target}`);
        // Reload price alerts if view is open
        if (showPriceAlerts) {
          loadPriceAlerts();
        }
      } catch (error) {
        console.error('Error creating price alert:', error);
        if (error.response?.status === 409) {
          alert('Price alert already exists for this product!');
        } else if (error.response?.status === 401) {
          alert('Please login again to set price alerts');
          logout();
        } else {
          alert('Failed to create price alert. Please try again.');
        }
      }
    }
  };

  const removePriceAlert = async (alertId) => {
    try {
      const token = localStorage.getItem('token');
      await axios.delete(`/api/price-alerts/${alertId}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      setPriceAlerts(priceAlerts.filter(alert => alert.id !== alertId));
      alert('🗑️ Price alert removed!');
    } catch (error) {
      console.error('Error removing price alert:', error);
      if (error.response?.status === 401) {
        alert('Please login again');
        logout();
      }
    }
  };

  const loadFavorites = async () => {
    setFavoritesLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('/api/favorites', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      setFavorites(response.data);
    } catch (error) {
      console.error('Error loading favorites:', error);
      if (error.response?.status === 401) {
        // Token expired, user needs to login again
        logout();
      }
    } finally {
      setFavoritesLoading(false);
    }
  };

  const loadPriceAlerts = async () => {
    setAlertsLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('/api/price-alerts', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      setPriceAlerts(response.data);
    } catch (error) {
      console.error('Error loading price alerts:', error);
      if (error.response?.status === 401) {
        // Token expired, user needs to login again
        logout();
      }
    } finally {
      setAlertsLoading(false);
    }
  };

  const handleShowFavorites = () => {
    setShowFavorites(!showFavorites);
    setShowPriceAlerts(false); // Hide alerts when showing favorites
    // Data is already loaded via useEffect, just refresh to get latest
    if (!showFavorites) {
      loadFavorites();
    }
  };

  const handleShowPriceAlerts = () => {
    setShowPriceAlerts(!showPriceAlerts);
    setShowFavorites(false); // Hide favorites when showing alerts
    // Data is already loaded via useEffect, just refresh to get latest
    if (!showPriceAlerts) {
      loadPriceAlerts();
    }
  };

  const removeFavorite = async (favoriteId) => {
    try {
      const token = localStorage.getItem('token');
      await axios.delete(`/api/favorites/${favoriteId}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      setFavorites(favorites.filter(fav => fav.id !== favoriteId));
      alert('Removed from favorites!');
    } catch (error) {
      console.error('Error removing favorite:', error);
      if (error.response?.status === 401) {
        alert('Please login again');
        logout();
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
            <button onClick={handleShowPriceAlerts} className="alerts-btn">
              🔔 Alerts ({priceAlerts.length})
            </button>
            <button onClick={handleShowFavorites} className="favorites-btn">
              ❤️ Favorites ({favorites.length})
            </button>
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
      {results.length > 0 && !showFavorites && !showPriceAlerts && (
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

      {/* Price Alerts Section */}
      {showPriceAlerts && (
        <section className="results-section">
          <div className="results-header">
            <h3>🔔 Your Price Alerts ({priceAlerts.length} active)</h3>
            <button onClick={handleShowPriceAlerts} className="close-favorites-btn">
              ✕ Close Alerts
            </button>
          </div>
          
          {alertsLoading && <div className="loading">Loading price alerts...</div>}
          
          {!alertsLoading && priceAlerts.length === 0 && (
            <div className="no-favorites">
              <p>🔔 No price alerts set yet!</p>
              <p>Search for products and set alerts to get notified when prices drop.</p>
            </div>
          )}
          
          {!alertsLoading && priceAlerts.length > 0 && (
            <div className="products-grid">
              {priceAlerts.map((alert) => (
                <div key={alert.id} className="product-card alert-card">
                  <h3>{alert.product_name}</h3>
                  <div className="alert-info">
                    <p className="target-price">🎯 Target Price: <strong>${alert.target_price}</strong></p>
                    <p className="status">Status: <span className={`status-${alert.is_active ? 'active' : 'inactive'}`}>
                      {alert.is_active ? '🟢 Active' : '🔴 Inactive'}
                    </span></p>
                    <p className="created">Created: {new Date(alert.created_at).toLocaleDateString()}</p>
                    {alert.last_checked && (
                      <p className="last-checked">Last checked: {new Date(alert.last_checked).toLocaleString()}</p>
                    )}
                  </div>
                  <div className="card-actions">
                    {alert.product_url && (
                      <a href={alert.product_url} target="_blank" rel="noopener noreferrer" className="view-btn">
                        Check Current Price
                      </a>
                    )}
                    <button onClick={() => removePriceAlert(alert.id)} className="remove-btn">
                      🗑️ Remove Alert
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {/* Favorites Section */}
      {showFavorites && (
        <section className="results-section">
          <div className="results-header">
            <h3>❤️ Your Favorites ({favorites.length} items)</h3>
            <button onClick={handleShowFavorites} className="close-favorites-btn">
              ✕ Close Favorites
            </button>
          </div>
          
          {favoritesLoading && <div className="loading">Loading favorites...</div>}
          
          {!favoritesLoading && favorites.length === 0 && (
            <div className="no-favorites">
              <p>No favorites yet! Start adding products you like.</p>
            </div>
          )}
          
          {!favoritesLoading && favorites.length > 0 && (
            <div className="products-grid">
              {favorites.map((favorite) => (
                <div key={favorite.id} className="product-card favorite-card">
                  <h3>{favorite.product_name}</h3>
                  <p className="price">${favorite.price}</p>
                  <p className="platform">Platform: {favorite.platform}</p>
                  <p className="date">Added: {new Date(favorite.added_at).toLocaleDateString()}</p>
                  <div className="card-actions">
                    {favorite.product_url && (
                      <a href={favorite.product_url} target="_blank" rel="noopener noreferrer" className="view-btn">
                        View Product
                      </a>
                    )}
                    <button onClick={() => removeFavorite(favorite.id)} className="remove-btn">
                      🗑️ Remove
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
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
  const { isAuthenticated, initializing } = useAuth();
  
  if (initializing) {
    return (
      <div className="App">
        <div className="loading-screen">
          <img src={logo} alt="AI Shopping Agent" className="loading-logo" />
          <h2>🛒 AI Shopping Agent</h2>
          <p>Loading your shopping experience...</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="App">
      {isAuthenticated ? <ShoppingApp /> : <LoginForm />}
    </div>
  );
}

export default App;
