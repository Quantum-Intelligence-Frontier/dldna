import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic block for ResNet
    Consists of two 3x3 convolutional layers
    """
    expansion = 1  # Output channel expansion factor

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

        # Skip connection (adjust dimensions with 1x1 convolution if necessary)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                         kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add skip connection
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    """Bottleneck block for ResNet
    Consists of a combination of 1x1, 3x3, and 1x1 convolutions
    """
    expansion = 4  # Output channel expansion factor

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 1x1 convolution to reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 convolution to increase channels
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                         kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # Add skip connection
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """ResNet model
    Args:
        block: BasicBlock or Bottleneck
        num_blocks: Number of blocks per stage
        num_classes: Number of classes to classify
        in_channels: Number of input channels (default: 3)
    """
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super().__init__()
        self.in_channels = 64

        # Modify the input channel of the first convolutional layer to in_channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # The rest of the code is the same
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Create a ResNet stage
        Args:
            block: Block type (BasicBlock or Bottleneck)
            out_channels: Number of output channels
            num_blocks: Number of blocks
            stride: Stride of the first block
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        # ResNet stages
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Classifier
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def ResNet18(num_classes=10, in_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)

def ResNet34(num_classes=10, in_channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, in_channels)

def ResNet50(num_classes=10, in_channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, in_channels)

def ResNet101(num_classes=10, in_channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, in_channels)

def ResNet152(num_classes=10, in_channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, in_channels)