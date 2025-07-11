import React, { useState } from 'react';
import './ProductCard.css';

const ProductCard = ({ product, onAddToFavorites, onPriceAlert }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [isFavorited, setIsFavorited] = useState(false);

  const handleAddToFavorites = async () => {
    setIsLoading(true);
    try {
      await onAddToFavorites(product);
      setIsFavorited(true);
    } catch (error) {
      console.error('Error adding to favorites:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(price);
  };

  const getPlatformIcon = (platform) => {
    const icons = {
      'Amazon': '🛒',
      'eBay': '🏪',
      'Walmart': '🛍️'
    };
    return icons[platform] || '🛒';
  };

  const getRatingStars = (rating) => {
    const stars = [];
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;
    
    for (let i = 0; i < fullStars; i++) {
      stars.push(<span key={i} className="star full">★</span>);
    }
    
    if (hasHalfStar) {
      stars.push(<span key="half" className="star half">★</span>);
    }
    
    const emptyStars = 5 - Math.ceil(rating);
    for (let i = 0; i < emptyStars; i++) {
      stars.push(<span key={`empty-${i}`} className="star empty">☆</span>);
    }
    
    return stars;
  };

  return (
    <div className="product-card">
      <div className="product-image-container">
        {product.image && (
          <img 
            src={product.image} 
            alt={product.name}
            className="product-image"
            loading="lazy"
          />
        )}
        <div className="platform-badge">
          {getPlatformIcon(product.platform)} {product.platform}
        </div>
      </div>
      
      <div className="product-info">
        <h3 className="product-name" title={product.name}>
          {product.name.length > 60 
            ? `${product.name.substring(0, 60)}...` 
            : product.name
          }
        </h3>
        
        <div className="product-price">
          {formatPrice(product.price)}
        </div>
        
        {product.rating > 0 && (
          <div className="product-rating">
            <div className="stars">
              {getRatingStars(product.rating)}
            </div>
            <span className="rating-text">({product.rating.toFixed(1)})</span>
          </div>
        )}
        
        <div className="product-actions">
          <a 
            href={product.url} 
            target="_blank" 
            rel="noopener noreferrer"
            className="btn btn-primary"
          >
            View Product
          </a>
          
          <button 
            onClick={handleAddToFavorites}
            className={`btn btn-secondary ${isFavorited ? 'favorited' : ''}`}
            disabled={isLoading}
          >
            {isLoading ? '...' : isFavorited ? '❤️' : '🤍'}
          </button>
          
          <button 
            onClick={() => onPriceAlert(product)}
            className="btn btn-outline"
            title="Set Price Alert"
          >
            🔔
          </button>
        </div>
      </div>
    </div>
  );
};

export default ProductCard;
