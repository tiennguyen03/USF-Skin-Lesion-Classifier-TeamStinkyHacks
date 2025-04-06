import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './ResultsPage.css';

interface Diagnosis {
  condition: string;
  confidence: number;
  recommendations: string[];
}

interface LocationState {
  imageUri: string;
  diagnosis: Diagnosis;
}

const ResultsPage: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { imageUri, diagnosis } = location.state as LocationState;

  return (
    <div className="results-container">
      <div className="results-content">
        <h1 className="results-title">Diagnosis Results</h1>

        <div className="image-container">
          <img src={imageUri} alt="Skin lesion" className="lesion-image" />
        </div>

        <div className="diagnosis-container">
          <h2 className="diagnosis-title">Diagnosis</h2>
          <p className="condition-text">{diagnosis.condition}</p>
          <p className="confidence-text">
            Confidence: {(diagnosis.confidence * 100).toFixed(1)}%
          </p>

          <h2 className="recommendations-title">Recommendations</h2>
          <ul className="recommendations-list">
            {diagnosis.recommendations.map((recommendation, index) => (
              <li key={index} className="recommendation-item">
                {recommendation}
              </li>
            ))}
          </ul>
        </div>

        <button
          className="back-button"
          onClick={() => navigate('/')}
        >
          Back to Home
        </button>
      </div>
    </div>
  );
};

export default ResultsPage; 