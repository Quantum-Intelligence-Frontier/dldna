import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """2D Convolution Layer
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (number of filters)
            kernel_size: Kernel size (integer or tuple)
            stride: Stride size (default: 1)
            padding: Padding size (default: 0)
        """
        super().__init__()
        # Convert kernel size to a tuple (if it's an integer, assume a square kernel)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        
        # Initialize trainable parameters (Xavier/Glorot initialization)
        # 1. Create a random tensor with a normal distribution (mean 0, std 1) using torch.randn()
        # 2. Xavier/Glorot scaling: Divide the generated weights by sqrt(number of input channels × kernel_size^2)
        # 3. This helps ensure that the output of each layer has an appropriate range, preventing vanishing/exploding gradients
        # shape: (output channels, input channels, kernel height, kernel width)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *self.kernel_size) / 
            np.sqrt(in_channels * self.kernel_size[0] * self.kernel_size[1]) # Corrected kernel_size usage
        )
        # Initialize bias
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x):
        # x shape: (batch size, input channels, height, width)
        batch_size, in_channels, in_height, in_width = x.shape
        
        # Apply padding
        if self.padding > 0:
            x = F.pad(x, [self.padding] * 4)  # Same padding on all sides (top, bottom, left, right)
            
        # Calculate the size of the output feature map
        # (W - K + 2P) / S + 1,  W: input size, K: kernel size, P: padding, S: stride
        out_height = (in_height + 2 * self.padding - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size[1]) // self.stride[1] + 1
        
        # Initialize the output tensor
        out = torch.zeros(batch_size, self.weight.shape[0], out_height, out_width)
        
        # Perform convolution operation
        for b in range(batch_size):  # For each batch
            for c_out in range(self.weight.shape[0]):  # For each output channel
                for h in range(out_height):  # For output height
                    for w in range(out_width):  # For output width
                        # Calculate the input region at the current position
                        h_start = h * self.stride[0]
                        w_start = w * self.stride[1]
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]
                        
                        # Extract the receptive field
                        receptive_field = x[b, :, h_start:h_end, w_start:w_end]
                        # Convolution operation: (sum of input * weight) + bias
                        out[b, c_out, h, w] = (receptive_field * self.weight[c_out]).sum() + self.bias[c_out]
        
        return out

class SimpleMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None):
        """2D Max Pooling Layer
        Args:
            kernel_size: Pooling window size
            stride: Stride (default: same as kernel_size)
        """
        super().__init__()
        # Convert kernel size to a tuple
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        # If stride is not specified, set it equal to kernel_size
        self.stride = stride if stride is not None else self.kernel_size
        
    def forward(self, x):
        # x shape: (batch size, channels, height, width)
        batch_size, channels, height, width = x.shape
        
        # Calculate the size of the output feature map
        out_height = (height - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width - self.kernel_size[1]) // self.stride[1] + 1
        
        # Initialize the output tensor
        out = torch.zeros(batch_size, channels, out_height, out_width)
        
        # Perform max pooling operation
        for b in range(batch_size):  # For each batch
            for c in range(channels):  # For each channel
                for h in range(out_height):  # For output height
                    for w in range(out_width):  # For output width
                        # Calculate the position of the current pooling window
                        h_start = h * self.stride[0]
                        w_start = w * self.stride[1]
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]
                        
                        # Select the maximum value in the current window
                        out[b, c, h, w] = x[b, c, h_start:h_end, w_start:w_end].max()
        
        return out