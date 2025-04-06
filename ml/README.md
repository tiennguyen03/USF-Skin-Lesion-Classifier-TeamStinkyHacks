# Skin Lesion Classification Web App

This is a web application for classifying skin lesion images as either benign or malignant using a deep learning model based on EfficientNet.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have the trained model checkpoint file (`model_checkpoint_epoch_20.pth`) in the same directory as the app.

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Upload a skin lesion image and click "Analyze Image" to get the classification results.

## Project Structure

- `app.py`: Flask web application
- `model.py`: Neural network model architecture
- `templates/index.html`: Web interface template
- `requirements.txt`: Python dependencies
- `model_checkpoint_epoch_20.pth`: Trained model weights

## Model Architecture

The model uses EfficientNet-B0 as the backbone and includes:
- Image feature extraction using EfficientNet
- Optional metadata processing (if provided)
- Multi-layer classifier with dropout for regularization

## Performance

The model achieves:
- Overall accuracy: 96.25%
- Strong performance on both benign and malignant classes
- Balanced precision and recall metrics

## Notes

- The model expects RGB images and will automatically resize them to 224x224 pixels
- Input images are normalized to the range [0, 1]
- The model outputs probabilities for both benign and malignant classes 