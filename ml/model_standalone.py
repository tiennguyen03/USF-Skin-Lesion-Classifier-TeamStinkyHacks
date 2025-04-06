import torch
import torch.nn as nn
import torchvision.models as models

class SkinLesionModel(nn.Module):
    def __init__(self, image_size=224, metadata_dim=0, num_classes=2):
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