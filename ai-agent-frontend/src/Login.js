// Login.js
import React, { useState } from 'react';
import './login.css'; // Import the CSS file for styling

function Login({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleLogin = (e) => {
    e.preventDefault();
    if (username === 'user' && password === 'password') {
      onLogin(username);
    } else {
      setError('Invalid credentials');
    }
  };

  return (
    <div className="login-container">
      <img src="./images/logo.jpg" alt="Logo" className="logo" /> {/* Update the path to your logo */}
      <h2>Login</h2>
      
      <form onSubmit={handleLogin}>
        <div className="form-group">
          <label>Username:</label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </div>
        <div className="form-group">
          <label>Password:</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        {error && <p className="error-message">{error}</p>}
        <button type="submit">Login</button>
        
      </form>
    </div>
  );
}

export default Login;
