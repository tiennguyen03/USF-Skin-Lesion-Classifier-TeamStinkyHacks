import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from train_model import SkinLesionModel, load_and_preprocess_data
from torch.utils.data import DataLoader, TensorDataset

# Configuration
CONFIG = {
    'image_size': 224,
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 0.001,
    'metadata_dim': 0,  # No metadata for now
    'num_classes': 2  # Binary classification: Benign vs Malignant
}

def evaluate_model(model_path, data_dir):
    # Load the model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SkinLesionModel(
        image_size=CONFIG['image_size'],
        metadata_dim=CONFIG['metadata_dim'],
        num_classes=CONFIG['num_classes']
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load the data
    images, metadata, labels = load_and_preprocess_data(data_dir, None)
    
    # Create dataset and dataloader
    images = torch.FloatTensor(images)
    labels = torch.LongTensor(labels)
    if metadata is not None:
        metadata = torch.FloatTensor(metadata)
        dataset = TensorDataset(images, metadata, labels)
    else:
        dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # Initialize lists to store predictions
    all_predicted = []
    all_labels = []
    
    # Process in batches
    with torch.no_grad():
        for batch in dataloader:
            if metadata is not None:
                batch_images, batch_metadata, batch_labels = batch
                batch_images = batch_images.to(device)
                batch_metadata = batch_metadata.to(device)
            else:
                batch_images, batch_labels = batch
                batch_images = batch_images.to(device)
                batch_metadata = None
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_images, batch_metadata)
            _, predicted = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            
            # Clear memory
            del batch_images, outputs, predicted
            if batch_metadata is not None:
                del batch_metadata
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Convert to numpy arrays
    all_predicted = np.array(all_predicted)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy
    accuracy = (all_predicted == all_labels).mean()
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predicted)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predicted,
                              target_names=['Benign', 'Malignant']))

    # Save some example predictions
    num_examples = 5
    indices = np.random.choice(len(images), num_examples, replace=False)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_examples, i + 1)
        img = images[idx].numpy().transpose(1, 2, 0)
        plt.imshow(img)
        plt.title(f'True: {"Benign" if all_labels[idx] == 0 else "Malignant"}\n'
                 f'Pred: {"Benign" if all_predicted[idx] == 0 else "Malignant"}')
        plt.axis('off')
    plt.savefig('example_predictions.png')
    plt.close()

if __name__ == "__main__":
    # Use the latest checkpoint
    model_path = "model_checkpoint_epoch_20.pth"  # Latest checkpoint
    data_dir = "../data/train"
    
    evaluate_model(model_path, data_dir) 