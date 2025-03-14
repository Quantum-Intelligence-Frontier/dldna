import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for MNIST classification.
    This model consists of two convolutional layers, a dropout layer, and two
    fully connected layers.  It includes methods for weight initialization and
    for extracting intermediate feature representations for visualization.
    """
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)    # Input: 28x28x1, Output: 26x26x16
        self.conv2 = nn.Conv2d(16, 32, 3)   # Input: 13x13x16, Output: 11x11x32
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(32 * 5 * 5, 32)  # Fully connected layer
        self.fc2 = nn.Linear(32, 10)        # Output layer (10 classes)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the weights of the convolutional and linear layers.
        - Conv layers: Kaiming He initialization (fan_out mode, ReLU nonlinearity).
        - Linear layers: Normal distribution (mean=0, std=0.01).
        - Biases: Initialized to zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Defines the forward pass of the network.  Stores intermediate activations.
        """
        batch_size = x.size(0)
        self.feat1 = F.relu(self.conv1(x))  # Store for visualization
        x = F.max_pool2d(self.feat1, 2)     # Output: 13x13x16
        self.feat2 = F.relu(self.conv2(x))  # Store for visualization
        x = F.max_pool2d(self.feat2, 2)     # Output: 5x5x32
        x = x.view(batch_size, -1)          # Flatten
        self.feat3 = F.relu(self.fc1(x))    # Store for visualization
        x = self.dropout(self.feat3)
        x = self.fc2(x)
        return x

    def get_embeddings(self, x):
        """
        Extracts the embeddings from the penultimate layer (fc1).
        """
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(batch_size, -1)
        return F.relu(self.fc1(x))  # Return the activations of fc1

    def get_features(self):
        """Returns the intermediate feature maps for visualization."""
        return self.feat1, self.feat2, self.feat3