import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import './ScanPage.css';

const ScanPage: React.FC = () => {
  const navigate = useNavigate();
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setIsProcessing(true);
      setError(null);

      // Create a preview URL for the image
      const imageUrl = URL.createObjectURL(file);

      // In a real app, we would send the image to our backend for processing
      // For now, we'll simulate a delay and navigate to results
      setTimeout(() => {
        setIsProcessing(false);
        navigate('/results', {
          state: {
            imageUri: imageUrl,
            diagnosis: {
              condition: 'Melanoma',
              confidence: 0.85,
              recommendations: [
                'Schedule a dermatologist appointment',
                'Monitor for changes in size and color',
                'Protect from sun exposure',
              ],
            },
          },
        });
      }, 2000);
    } catch (err) {
      setError('Failed to process image');
      setIsProcessing(false);
    }
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="scan-container">
      <div className="scan-content">
        <h1 className="scan-title">Scan Skin Lesion</h1>
        <p className="scan-instructions">
          Upload a clear photo of the skin lesion for analysis
        </p>

        <div className="upload-area" onClick={handleButtonClick}>
          {isProcessing ? (
            <div className="processing">
              <div className="spinner"></div>
              <p>Processing image...</p>
            </div>
          ) : (
            <>
              <div className="upload-icon">📷</div>
              <p>Click to upload or drag and drop</p>
            </>
          )}
        </div>

        {error && <p className="error-message">{error}</p>}

        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="image/*"
          className="file-input"
        />
      </div>
    </div>
  );
};

export default ScanPage; 