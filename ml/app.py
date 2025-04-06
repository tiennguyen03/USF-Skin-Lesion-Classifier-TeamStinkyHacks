import os
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model_standalone import SkinLesionModel

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

# Configuration
config = {
    'image_size': 224,
    'metadata_dim': 0,
    'num_classes': 2
}

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model = SkinLesionModel(
    image_size=config['image_size'],
    metadata_dim=config['metadata_dim'],
    num_classes=config['num_classes']
).to(device)

# Load the model weights
checkpoint = torch.load('model_checkpoint_epoch_20.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def process_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((config['image_size'], config['image_size']))
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.transpose(image, (2, 0, 1))  # Change to (C, H, W)
    image = torch.FloatTensor(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        
    return prediction, probabilities[0].cpu().numpy()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Save the uploaded file temporarily
    temp_path = 'temp_upload.jpg'
    file.save(temp_path)
    
    try:
        # Process the image and get prediction
        prediction, probabilities = process_image(temp_path)
        
        # Clean up the temporary file
        os.remove(temp_path)
        
        # Prepare the response
        result = {
            'prediction': 'Benign' if prediction == 0 else 'Malignant',
            'benign_probability': f'{probabilities[0]:.2%}',
            'malignant_probability': f'{probabilities[1]:.2%}'
        }
        
        return jsonify(result)
    
    except Exception as e:
        # Clean up the temporary file in case of error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001) 