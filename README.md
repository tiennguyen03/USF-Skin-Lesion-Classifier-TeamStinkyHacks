# SkinLesionAI - Mobile Skin Lesion Detection App

A mobile application that uses computer vision and machine learning to detect and diagnose skin lesions, moles, and blemishes.

## Project Structure

```
├── mobile/           # React Native mobile application
├── ml/              # Machine learning model training and inference
├── api/             # Backend API server
└── data/            # Training and testing datasets
```

## Features

- Real-time skin lesion detection using device camera
- AI-powered diagnosis of detected lesions
- Historical tracking of skin changes
- Educational information about different types of skin conditions

## Technology Stack

- Frontend: React Native
- Backend: Python FastAPI
- ML Framework: TensorFlow/PyTorch
- Database: SQLite (mobile) / PostgreSQL (backend)

## Setup Instructions

1. Install dependencies:
   ```bash
   # Mobile app
   cd mobile
   npm install

   # Backend
   cd api
   pip install -r requirements.txt

   # ML
   cd ml
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   # Mobile app
   cd mobile
   npm run android  # or npm run ios

   # Backend
   cd api
   uvicorn main:app --reload
   ```

## Data Sources

The application uses the ISIC (International Skin Imaging Collaboration) dataset for training the machine learning model.

## License

MIT License
