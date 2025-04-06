import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Configuration Constants
IMG_SIZE = 224  # Target size for model input
DATA_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
DATA_STD = [0.229, 0.224, 0.225]   # ImageNet std

class SkinCancerDataset(Dataset):
    """Optimized dataset loader for skin cancer JPG images"""
    
    def __init__(self, root_dir, mode='train'):
        """
        Args:
            root_dir (str): Path containing 'benign' and 'malignant' folders
            mode (str): 'train' or 'test' (controls augmentations)
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = self._get_transforms()
        self.image_paths = []
        self.labels = []
        
        self._load_jpgs()  # Load all JPG files
        
        if not self.image_paths:
            raise ValueError(f"No JPGs found in {root_dir}/{{benign,malignant}}")

    def _load_jpgs(self):
        """Efficiently loads JPG files with progress tracking"""
        for label_value, label_name in enumerate(['benign', 'malignant']):
            label_dir = os.path.join(self.root_dir, label_name)
            if not os.path.exists(label_dir):
                continue
                
            # Get all JPG files (case insensitive)
            jpg_files = [f for f in os.listdir(label_dir) 
                        if f.lower().endswith('.jpg')]
            
            # Load with progress bar
            for img_name in tqdm(jpg_files, 
                               desc=f"Loading {label_name} images",
                               unit='img'):
                self.image_paths.append(os.path.join(label_dir, img_name))
                self.labels.append(label_value)

    def _get_transforms(self):
        """Builds appropriate transforms for train/test modes"""
        if self.mode == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(DATA_MEAN, DATA_STD)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(DATA_MEAN, DATA_STD)
            ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            with Image.open(self.image_paths[idx]) as img:
                return self.transform(img.convert('RGB')), \
                       torch.tensor(self.labels[idx], dtype=torch.long)
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {str(e)}")
            # Return next valid image if current fails
            return self[(idx + 1) % len(self)]

def get_dataloaders(train_dir, test_dir, batch_size=32):
    """
    Creates train and test dataloaders
    
    Args:
        train_dir (str): Path to training images
        test_dir (str): Path to test images
        batch_size (int): Images per batch
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_set = SkinCancerDataset(train_dir, 'train')
    test_set = SkinCancerDataset(test_dir, 'test')
    
    return (
        DataLoader(train_set, 
                 batch_size=batch_size, 
                 shuffle=True,
                 num_workers=4,
                 pin_memory=True),
        DataLoader(test_set,
                 batch_size=batch_size,
                 shuffle=False,
                 num_workers=2)
    )