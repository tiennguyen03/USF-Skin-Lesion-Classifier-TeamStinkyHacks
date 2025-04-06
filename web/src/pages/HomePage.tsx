import React from 'react';
import { Link } from 'react-router-dom';
import './HomePage.css';

const HomePage: React.FC = () => {
  return (
    <div className="home-container">
      <div className="header">
        <h1 className="title">Skin Lesion AI</h1>
        <p className="subtitle">
          Detect and analyze skin lesions using AI
        </p>
      </div>

      <div className="content">
        <Link to="/scan" className="button">
          <div className="button-icon">📷</div>
          <span className="button-text">Scan Lesion</span>
        </Link>

        <Link to="/history" className="button">
          <div className="button-icon">📋</div>
          <span className="button-text">View History</span>
        </Link>
      </div>
    </div>
  );
};

export default HomePage; 