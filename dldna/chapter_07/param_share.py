import torch
import torch.nn as nn
import pandas as pd

def compare_parameter_counts(input_sizes):
    """Compares the number of parameters in CNN and FC layers for various input sizes."""
    
    results = []
    for size in input_sizes:
        # CNN layer (3x3 kernel, single channel)
        conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        conv_params = sum(p.numel() for p in conv.parameters())
        
        # FC layer
        fc = nn.Linear(size * size, size * size)
        fc_params = sum(p.numel() for p in fc.parameters())
        
        results.append({
            'Input Size': f"{size}x{size}",
            'Conv Params': conv_params,
            'FC Params': fc_params,
            'Ratio (FC/Conv)': fc_params / conv_params
        })
    
    return pd.DataFrame(results)

def show_example(size=32):
    """Shows an example of parameter count comparison for a specific input size."""
    print(f"\nExample with {size}x{size} input:")
    
    # 3x3 convolution layer
    conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    conv_params = sum(p.numel() for p in conv.parameters())
    
    # FC layer of the same size
    fc = nn.Linear(size * size, size * size)
    fc_params = sum(p.numel() for p in fc.parameters())
    
    print(f"CNN parameters: {conv_params:,} (fixed)")
    print(f"FC parameters: {fc_params:,}")
    print(f"Parameter reduction: {((fc_params - conv_params) / fc_params * 100):.4f}%")