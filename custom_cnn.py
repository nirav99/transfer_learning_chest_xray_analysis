import torch
import torch.nn as nn

class custom_cnn(nn.Module):
    def __init__(self, img_size):
        super(custom_cnn, self).__init__()
        self.img_size = img_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout Layer 
        dropout_prob = 0.5
        self.dropout = nn.Dropout(dropout_prob)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * (img_size // 4) * (img_size // 4), 64)  # Increased size for more capacity
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Output layer for binary classification

        # Batch Normalization after Fully Connected Layers
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(32)


    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        # First Fully Connected Layer + Batch Normalization + Dropout + ReLU
        x = torch.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)

        # Second Fully Connected Layer + Batch Normalization + Dropout + ReLU
        x = torch.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)

        # Output layer (no activation needed here for binary classification)
        x = self.fc3(x)

        return x
