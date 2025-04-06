import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './HistoryPage.css';

interface ScanHistory {
  id: string;
  date: string;
  imageUri: string;
  diagnosis: {
    condition: string;
    confidence: number;
  };
}

const HistoryPage: React.FC = () => {
  const navigate = useNavigate();
  const [scanHistory, setScanHistory] = useState<ScanHistory[]>([]);

  useEffect(() => {
    // In a real app, this would fetch from a backend
    // For now, we'll use mock data
    const mockHistory: ScanHistory[] = [
      {
        id: '1',
        date: '2024-03-15',
        imageUri: 'https://via.placeholder.com/150',
        diagnosis: {
          condition: 'Melanoma',
          confidence: 0.85
        }
      },
      {
        id: '2',
        date: '2024-03-14',
        imageUri: 'https://via.placeholder.com/150',
        diagnosis: {
          condition: 'Benign keratosis',
          confidence: 0.92
        }
      }
    ];
    setScanHistory(mockHistory);
  }, []);

  const handleScanClick = (scan: ScanHistory) => {
    navigate('/results', {
      state: {
        imageUri: scan.imageUri,
        diagnosis: {
          condition: scan.diagnosis.condition,
          confidence: scan.diagnosis.confidence,
          recommendations: [
            'Schedule a follow-up appointment with your dermatologist',
            'Monitor the lesion for any changes',
            'Protect the area from sun exposure'
          ]
        }
      }
    });
  };

  return (
    <div className="history-container">
      <div className="history-content">
        <h1 className="history-title">Scan History</h1>

        {scanHistory.length === 0 ? (
          <p className="no-history">No scan history available</p>
        ) : (
          <div className="history-list">
            {scanHistory.map((scan) => (
              <div
                key={scan.id}
                className="history-item"
                onClick={() => handleScanClick(scan)}
              >
                <img
                  src={scan.imageUri}
                  alt="Skin lesion"
                  className="history-image"
                />
                <div className="history-details">
                  <p className="history-date">{scan.date}</p>
                  <p className="history-condition">
                    {scan.diagnosis.condition}
                  </p>
                  <p className="history-confidence">
                    Confidence: {(scan.diagnosis.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}

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

export default HistoryPage; 