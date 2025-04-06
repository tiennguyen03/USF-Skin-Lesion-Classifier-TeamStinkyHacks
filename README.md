# USF Skin Lesion Classifier - Team StinkyHacks

A web and mobile application that uses deep learning to classify skin lesions and provide instant diagnosis recommendations.

## Project Structure

```
├── ml/              # Machine learning model and Flask server
│   ├── app.py      # Flask server for model inference
│   ├── model_standalone.py  # Model architecture and inference code
│   └── templates/  # Web interface templates
├── web/            # React web application
│   └── src/        # React source code
└── mobile/         # React Native mobile application
    └── src/        # Mobile app source code
```

## Features

- Real-time skin lesion classification using device camera
- Web and mobile interfaces for easy access
- Instant diagnosis recommendations
- Historical tracking of skin changes
- Educational information about different types of skin conditions

## Technology Stack

- Frontend: React (Web) / React Native (Mobile)
- Backend: Python Flask
- ML Framework: PyTorch
- Model: ResNet50-based architecture

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/tiennguyen03/USF-Skin-Lesion-Classifier-TeamStinkyHacks.git
cd USF-Skin-Lesion-Classifier-TeamStinkyHacks
```

### 2. Set Up the ML Backend

1. Create and activate a Python virtual environment:
```bash
cd ml
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Download the model checkpoint:
   - The model checkpoint file is hosted separately due to size limitations
   - Download link: [Google Drive Link]
   - Place the downloaded `model_checkpoint_epoch_20.pth` file in the `ml/` directory

4. Start the Flask server:
```bash
python app.py
```
The server will start at http://localhost:5001

### 3. Set Up the Web Interface

1. Install Node.js dependencies:
```bash
cd web
npm install
```

2. Start the development server:
```bash
npm start
```
The web interface will be available at http://localhost:3000

### 4. Set Up the Mobile App

1. Install Node.js dependencies:
```bash
cd mobile
npm install
```

2. Start the development server:
```bash
npm start
```

3. Run on your device:
```bash
npm run android  # For Android
# or
npm run ios     # For iOS
```

## Model Information

- Architecture: ResNet50-based CNN
- Training Dataset: ISIC 2019 Challenge Dataset
- Classes: 8 different types of skin lesions
- Accuracy: 96.25% on validation set

## API Endpoints

- POST `/predict`: Upload an image for classification
  - Input: Multipart form data with image file
  - Output: JSON with prediction results and confidence scores

## Contributors

### Team StinkyHacks
- **Tien Nguyen** - ML Development & Backend
- **Thomveebol Phorn** - ML Development & Backend
- **Andy Bertil** - ML Development & Backend
- **James Huynh** - ML & Computer Vision
- **Cem Tutar** - Frontend Development

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ISIC Dataset for providing the training data
- PyTorch team for the deep learning framework
- React and React Native communities for the frontend frameworks
