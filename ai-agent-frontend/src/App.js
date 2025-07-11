import React, { useState, useEffect } from "react";
import './App.css'; // Import the CSS file for styling
import logo from './images/logo.jpg'; // Assuming the logo is in the assets folder

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState('');
  const [criteria, setCriteria] = useState({ budget: "", product: "" });
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sortOption, setSortOption] = useState("price");
  const [filterText, setFilterText] = useState("");
  const [useNLP, setUseNLP] = useState(false);
  const [currency, setCurrency] = useState("USD");
  const [language, setLanguage] = useState("en");
  const [optimization, setOptimization] = useState("value");
  const [error, setError] = useState(null);
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [chatMessages, setChatMessages] = useState([]); // State for chat messages
  const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();

  const handleLogin = (name) => {
    setUsername(name);
    setIsLoggedIn(true);
  };

  const getGreeting = () => {
    const hour = new Date().getHours();
    let greeting = '';
    if (hour < 12) greeting = "Good morning";
    else if (hour < 18) greeting = "Good afternoon";
    else greeting = "Good evening";
    return `${greeting}, ${username}`;
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

  useEffect(() => {
    if (isLoggedIn) {
      const greetingMessage = getGreeting();
      const speech = new SpeechSynthesisUtterance(greetingMessage);
      window.speechSynthesis.speak(speech);
    }

    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onend = () => {
      setIsListening(false);
      if (criteria.budget && criteria.product) {
        document.querySelector('.submit-button').click();
      }
    };

    recognition.onresult = (event) => {
      const latestTranscript = event.results[event.results.length - 1][0].transcript;
      setTranscript(latestTranscript);

      const budgetMatch = latestTranscript.match(/\d+/);
      const productMatch = latestTranscript.replace("product", "").trim();

      if (budgetMatch) {
        setCriteria((prev) => ({ ...prev, budget: budgetMatch[0] }));
      }

      if (productMatch) {
        setCriteria((prev) => ({ ...prev, product: productMatch }));
      }

      if (budgetMatch && productMatch) {
        document.querySelector('.submit-button').click();
      }
    };

    return () => {
      recognition.stop();
    };
  }, [criteria, isLoggedIn]);

  const handleChange = (e) => {
    setCriteria({ ...criteria, [e.target.name]: e.target.value });
  };

  const handleSortChange = (e) => {
    setSortOption(e.target.value);
    setResults((prevResults) =>
      [...prevResults].sort((a, b) => (e.target.value === "price" ? a.price - b.price : a.name.localeCompare(b.name)))
    );
  };

  const handleFilterChange = (e) => {
    setFilterText(e.target.value);
  };

  const handleNLPChange = (e) => {
    setUseNLP(e.target.checked);
  };

  const handleCurrencyChange = (e) => {
    setCurrency(e.target.value);
  };

  const handleLanguageChange = (e) => {
    setLanguage(e.target.value);
  };

  const handleOptimizationChange = (e) => {
    setOptimization(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:5000/api/auth/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...criteria, use_nlp: useNLP, currency, language, optimization }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch recommendations.");
      }

      const data = await response.json();
      setResults(data.sort((a, b) => a.price - b.price));
    } catch (error) {
      console.error(error);
      setError("An error occurred while fetching recommendations.");
    } finally {
      setLoading(false);
    }
  };

  const filteredResults = results.filter((item) =>
    item.name.toLowerCase().includes(filterText.toLowerCase())
  );

  const startListening = () => {
    recognition.start();
  };

  const handleChatSubmit = async (message) => {
    setChatMessages((prev) => [...prev, { text: message, sender: "user" }]);
    try {
      const response = await fetch("http://localhost:5000/api/auth/chatgpt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: message }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response from ChatGPT.");
      }

      const data = await response.json();
      setChatMessages((prev) => [...prev, { text: data.chatgpt_response, sender: "chatgpt" }]);
    } catch (error) {
      console.error(error);
      setChatMessages((prev) => [...prev, { text: "Error: Unable to get response.", sender: "chatgpt" }]);
    }
  };

  if (!isLoggedIn) {
    return (
      <div className="login-container">
        <h1>Login</h1>
        <input
          type="text"
          placeholder="Enter your name"
          onChange={(e) => setUsername(e.target.value)}
        />
        <button onClick={() => handleLogin(username)}>Login</button>
      </div>
    );
  }

  return (
    <div className="app-container">
      <img src={logo} alt="Logo" className="logo" />
      <h1 className="title">AI Shopping Agent</h1>
      <h2>{getGreeting()}. {getAssistantMessage()}</h2>

      {error && <p className="error-message">{error}</p>}

      <form onSubmit={handleSubmit} className="form-container">
        <div className="form-group">
          <label>Budget:</label>
          <input
            type="number"
            name="budget"
            value={criteria.budget}
            onChange={handleChange}
            required
            className="input-field"
            aria-label="Enter your budget"
          />
        </div>
        
        <div className="form-group">
          <label>Product Type:</label>
          <input
            type="text"
            name="product"
            value={criteria.product}
            onChange={handleChange}
            required
            className="input-field"
            aria-label="Enter product type"
          />
        </div>
        <div className="form-group">
          <label>Optimization Criteria:</label>
          <select value={optimization} onChange={handleOptimizationChange} className="select-field" aria-label="Select optimization criteria">
            <option value="value">Maximum Value</option>
            <option value="price">Minimum Price</option>
            <option value="reviews">Best Reviews</option>
          </select>
        </div>
        <div className="form-group">
          <label>Use NLP for better results:</label>
          <input
            type="checkbox"
            checked={useNLP}
            onChange={handleNLPChange}
            className="checkbox"
            aria-label="Enable NLP for better results"
          />
        </div>
        <div className="form-group">
          <label>Currency:</label>
          <select value={currency} onChange={handleCurrencyChange} className="select-field" aria-label="Select currency">
            <option value="USD">USD</option>
            <option value="EUR">EUR</option>
            <option value="INR">INR</option>
          </select>
        </div>
        <div className="form-group">
          <label>Language:</label>
          <select value={language} onChange={handleLanguageChange} className="select-field" aria-label="Select language">
            <option value="en">English</option>
            <option value="es">Spanish</option>
            <option value="fr">French</option>
          </select>
        </div>
        <button
          type="button"
          onClick={startListening}
          className="voice-button"
          disabled={isListening}
          aria-label="Activate voice command for budget and product"
        >
          {isListening ? "Listening..." : "Voice Command for Budget and Product"}
        </button>
        <button type="submit" disabled={loading} className="submit-button" aria-label="Submit your request for recommendations">
          {loading ? "Loading..." : "Get Recommendations"}
        </button>
      </form>

      <div className="transcript-container">
        <h3>Voice Input:</h3>
        <p>{transcript}</p>
      </div>

      <div className="filters-container">
        <h2>Filters & Sorting</h2>
        <div className="filter-group">
          <label> Sort by:</label>
          <select value={sortOption} onChange={handleSortChange} className="select-field" aria-label="Sort results by">
            <option value="price">Price</option>
            <option value="name">Name</option>
          </select>
        </div>
        <div className="filter-group">
          <label>Filter by Name:</label>
          <input
            type="text"
            value={filterText}
            onChange={handleFilterChange}
            placeholder="Enter product name"
            className="input-field"
            aria-label="Filter results by name"
          />
        </div>
      </div>

      <h2>Recommendations:</h2>
      <div className="recommendations-container">
        {filteredResults.length === 0 ? (
          <p>No recommendations available.</p>
        ) : (
          filteredResults.map((item, index) => (
            <div key={index} className="recommendation-card">
              <a href={item.url} target="_blank" rel="noopener noreferrer" className="recommendation-link">
                <h3>{item.name}</h3>
                <p>Price: {currency === "INR" ? `₹${item.price * 75}` : currency === "EUR" ? `€${item.price * 0.9}` : `$${item.price}`}</p>
                <p>{item.description}</p>
              </a>
            </div>
          ))
        )}
      </div>

      <div className="chat-container">
        <h2>Chat with Assistant</h2>
        <div className="chat-messages">
          {chatMessages.map((message, index) => (
            <div key={index} className={message.sender === "user" ? "user-message" : "chatgpt-message"}>
              {message.text}
            </div>
          ))}
        </div>
        <button onClick={() => handleChatSubmit("Hi")} className="chat-button">Send Message</button>
      </div>
      <h5>Copyright © 2024 Design & Developed by Vinuka Dinethmin & Danuddika Jayasundara. All Rights Reserved.</h5>
    </div>
  );
}

export default App;
