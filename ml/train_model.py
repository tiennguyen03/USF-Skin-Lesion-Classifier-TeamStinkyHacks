import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, models
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 2  # Binary classification: malignant/benign
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
USE_METADATA = True  # Flag to control whether to use metadata features

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'train')
EXTRA_DIR = os.path.join(PROJECT_ROOT, 'data', 'extra')

# Class labels
CLASS_LABELS = ["benign", "malignant"]

class SkinLesionDataset(Dataset):
    def __init__(self, images, labels, metadata=None, transform=None):
        self.images = images
        self.labels = labels
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        if self.metadata is not None:
            return image, self.metadata[idx], label
        return image, label

class SkinLesionModel(nn.Module):
    def __init__(self, image_size, metadata_dim, num_classes):
        super(SkinLesionModel, self).__init__()
        
        # Image processing branch
        self.base_model = models.mobilenet_v2(pretrained=True)
        self.base_model.classifier = nn.Identity()  # Remove the final classification layer
        
        # Calculate the output size of the base model
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size, image_size)
            base_output_size = self.base_model(dummy_input).shape[1]
        
        # Metadata processing branch (if metadata is used)
        self.use_metadata = metadata_dim > 0
        if self.use_metadata:
            self.metadata_fc = nn.Sequential(
                nn.Linear(metadata_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            combined_size = base_output_size + 64
        else:
            combined_size = base_output_size
        
        # Combined processing
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, images, metadata=None):
        # Process images
        x = self.base_model(images)
        
        # Process metadata if available
        if self.use_metadata and metadata is not None:
            metadata_features = self.metadata_fc(metadata)
            x = torch.cat((x, metadata_features), dim=1)
        
        # Final classification
        x = self.classifier(x)
        return x

def create_model(metadata_dim=0):
    """Create a CNN model for skin lesion classification with optional metadata input."""
    # Base CNN model
    cnn_model = models.mobilenet_v2(pretrained=True)
    cnn_features = cnn_model.classifier[1].in_features
    
    # Create a new model that combines CNN features with metadata
    class CombinedModel(nn.Module):
        def __init__(self, cnn_model, cnn_features, metadata_dim, num_classes):
            super(CombinedModel, self).__init__()
            self.cnn_model = cnn_model
            self.cnn_model.classifier = nn.Identity()  # Remove the classifier
            
            # Calculate input dimension for the final classifier
            combined_features = cnn_features
            if metadata_dim > 0:
                combined_features += metadata_dim
            
            # Create a new classifier
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(combined_features, 512),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(512, num_classes)
            )
            
        def forward(self, x, metadata=None):
            # Get CNN features
            cnn_features = self.cnn_model(x)
            
            # Combine with metadata if available
            if metadata is not None:
                combined_features = torch.cat((cnn_features, metadata), dim=1)
            else:
                combined_features = cnn_features
                
            # Pass through classifier
            output = self.classifier(combined_features)
            return output
    
    # Create the combined model
    model = CombinedModel(cnn_model, cnn_features, metadata_dim, NUM_CLASSES)
    return model

def load_metadata():
    """Load and preprocess metadata from CSV files."""
    print("Loading metadata from extra directory...")
    
    # Load clinical data
    clinical_path = os.path.join(EXTRA_DIR, 'clinical.csv')
    if os.path.exists(clinical_path):
        clinical_df = pd.read_csv(clinical_path)
        print(f"Loaded clinical data with {len(clinical_df)} records")
    else:
        print("Clinical data not found")
        return None
    
    # Load pathology data
    pathology_path = os.path.join(EXTRA_DIR, 'pathology_detail.csv')
    if os.path.exists(pathology_path):
        pathology_df = pd.read_csv(pathology_path)
        print(f"Loaded pathology data with {len(pathology_df)} records")
    else:
        print("Pathology data not found")
        pathology_df = pd.DataFrame()
    
    # Merge data if both exist
    if not pathology_df.empty:
        # Assuming there's a common ID column to merge on
        # This is a placeholder - adjust based on actual column names
        try:
            merged_df = pd.merge(clinical_df, pathology_df, on='case_id', how='left')
            print(f"Merged data has {len(merged_df)} records")
        except Exception as e:
            print(f"Error merging data: {e}")
            merged_df = clinical_df
    else:
        merged_df = clinical_df
    
    # Select relevant features
    # This is a placeholder - adjust based on actual column names and domain knowledge
    feature_columns = []
    for col in merged_df.columns:
        # Skip non-numeric columns and ID columns
        if merged_df[col].dtype in [np.int64, np.float64] and 'id' not in col.lower():
            feature_columns.append(col)
    
    if not feature_columns:
        print("No suitable numeric features found in metadata")
        return None
    
    # Fill missing values with median
    for col in feature_columns:
        merged_df[col] = merged_df[col].fillna(merged_df[col].median())
    
    # Extract features
    metadata_features = merged_df[feature_columns].values
    print(f"Extracted {metadata_features.shape[1]} metadata features")
    
    return metadata_features, feature_columns

def load_and_preprocess_data(data_dir, extra_dir, image_size=224):
    """Load and preprocess the dataset"""
    print("\nLoading and preprocessing data...")
    
    print(f"\nLoading data from: {data_dir}")
    
    # Load metadata first
    print("Loading metadata from extra directory...")
    try:
        clinical_data = pd.read_csv(os.path.join(extra_dir, 'clinical.csv'))
        pathology_data = pd.read_csv(os.path.join(extra_dir, 'pathology.csv'))
        
        print(f"Loaded clinical data with {len(clinical_data)} records")
        print(f"Loaded pathology data with {len(pathology_data)} records")
        
        # Merge metadata
        try:
            metadata = pd.merge(clinical_data, pathology_data, on='case_id', how='left')
            print(f"Merged metadata has {len(metadata)} records")
        except Exception as e:
            print(f"Error merging data: {str(e)}")
            metadata = None
    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
        metadata = None
    
    # Initialize lists to store data
    images = []
    labels = []
    metadata_list = []
    
    # Process each class
    for class_name in ['benign', 'malignant']:
        class_dir = os.path.join(data_dir, class_name)
        print(f"\nLoading {class_name} images from {class_dir}")
        
        if not os.path.exists(class_dir):
            print(f"Directory not found: {class_dir}")
            continue
            
        # Get list of files
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(image_files)} images in {class_name} directory")
        
        for img_name in image_files:
            img_path = os.path.join(class_dir, img_name)
            
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img = img.resize((image_size, image_size))
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                # Transpose from (H, W, C) to (C, H, W)
                img_array = np.transpose(img_array, (2, 0, 1))
                
                # Extract case_id from filename (assuming format: case_id.jpg)
                case_id = os.path.splitext(img_name)[0]
                
                # Get metadata for this case if available
                if metadata is not None:
                    case_metadata = metadata[metadata['case_id'] == case_id]
                    if len(case_metadata) > 0:
                        # Select relevant features and convert to float32
                        metadata_features = case_metadata[['age', 'sex']].iloc[0]
                        # Convert categorical variables to numeric
                        metadata_features['sex'] = 1 if metadata_features['sex'] == 'male' else 0
                        metadata_features = metadata_features.astype(np.float32)
                    else:
                        # Use default values if metadata not found
                        metadata_features = pd.Series({'age': 0, 'sex': 0}, dtype=np.float32)
                    metadata_list.append(metadata_features)
                
                images.append(img_array)
                labels.append(1 if class_name == 'malignant' else 0)
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
    
    if not images:
        raise ValueError("No images were loaded. Please check the data directory structure.")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    metadata_array = np.array(metadata_list) if metadata_list else None
    
    print(f"\nTotal images loaded: {len(X)}")
    print(f"Class distribution: {np.bincount(y)}")
    if metadata_array is not None:
        print(f"Metadata shape: {metadata_array.shape}")
    
    return X, metadata_array, y

def train_model():
    """Train the model"""
    print("\nTraining model...")
    
    # Load and preprocess data
    X, metadata, y = load_and_preprocess_data(DATA_DIR, EXTRA_DIR)
    
    # Convert to tensors with explicit float32 dtype
    images = torch.FloatTensor(X).to(torch.float32)
    metadata = torch.FloatTensor(metadata).to(torch.float32) if metadata is not None else None
    labels = torch.LongTensor(y)
    
    # Get metadata dimension
    metadata_dim = metadata.shape[1] if metadata is not None else 0
    print(f"Metadata dimension: {metadata_dim}")
    
    # Create dataset and dataloader
    if metadata is not None:
        dataset = TensorDataset(images, metadata, labels)
    else:
        dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = SkinLesionModel(IMG_SIZE, metadata_dim, NUM_CLASSES)
    model = model.to(DEVICE)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device and ensure float32
            if metadata is not None:
                images, batch_metadata, labels = batch
                images = images.to(DEVICE).to(torch.float32)
                batch_metadata = batch_metadata.to(DEVICE).to(torch.float32)
            else:
                images, labels = batch
                images = images.to(DEVICE).to(torch.float32)
                batch_metadata = None
            labels = labels.to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, batch_metadata)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{EPOCHS} | Batch: {batch_idx}/{len(dataloader)} | '
                      f'Loss: {total_loss/(batch_idx+1):.4f} | '
                      f'Acc: {100.*correct/total:.2f}%')
        
        # Save model checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total
        }
        torch.save(checkpoint, f'model_checkpoint_epoch_{epoch+1}.pth')
    
    print("\nTraining completed!")

if __name__ == "__main__":
    train_model() 