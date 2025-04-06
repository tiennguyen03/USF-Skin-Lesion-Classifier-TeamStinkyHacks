import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import shutil

def download_dataset():
    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Create necessary directories
    os.makedirs('data/train/benign', exist_ok=True)
    os.makedirs('data/train/malignant', exist_ok=True)
    
    # Download the dataset
    print("Downloading dataset from Kaggle...")
    api.dataset_download_files('fanconic/skin-cancer-malignant-vs-benign', path='data/temp', unzip=True)
    
    # Move files to appropriate directories
    print("Organizing files...")
    source_dir = 'data/temp'
    
    # Move benign images
    benign_source = os.path.join(source_dir, 'benign')
    if os.path.exists(benign_source):
        for file in os.listdir(benign_source):
            if file.endswith('.jpg'):
                shutil.move(
                    os.path.join(benign_source, file),
                    os.path.join('data/train/benign', file)
                )
    
    # Move malignant images
    malignant_source = os.path.join(source_dir, 'malignant')
    if os.path.exists(malignant_source):
        for file in os.listdir(malignant_source):
            if file.endswith('.jpg'):
                shutil.move(
                    os.path.join(malignant_source, file),
                    os.path.join('data/train/malignant', file)
                )
    
    # Clean up temporary directory
    print("Cleaning up...")
    shutil.rmtree('data/temp')
    
    print("Dataset download and organization complete!")

if __name__ == "__main__":
    download_dataset() 