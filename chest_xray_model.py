import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class chest_xray_model(nn.Module):
    def __init__(self):
        super(chest_xray_model, self).__init__()

        # Load the pre-trained ResNet-18 model
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze all layers by setting requires_grad=False
        for param in self.model.parameters():
            param.requires_grad = False
   
        # Remove the last fully connected layer
        self.model.fc = nn.Identity()

        # Add convolutional layer
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        # batch normalization and dropout added to reduce overfitting
        self.batch_norm = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(0.2)

        # Add a global average pooling layer after the convolutions
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layer for binary classification
        self.fc = nn.Linear(256, 2)


    def forward(self, x):
        x = self.model(x)

#        print(f"Shape of output from resnet18 : {x.shape}")

        # Reshape the output to add spatial dimensions (1x1) for convolution
        x = x.unsqueeze(2).unsqueeze(3)  # Shape becomes (batch_size, 512, 1, 1)
#        print(f"Shape after unsqueezing : {x.shape}")

        # Pass through the convolutional layers with ReLU activation
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        
        # Apply global average pooling
        x = self.global_avg_pool(x)
        
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)
        
        # Pass through the final fully connected layer
        x = self.fc(x)
        
        return x
        
