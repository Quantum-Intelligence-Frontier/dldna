import torch
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self, act_func, input_shape=784, num_labels=10, 
                 hidden_shape=[256, 192, 128, 64], no_act=False):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = []
        layers_shape = [input_shape] + hidden_shape

        act_func_name = type(act_func).__name__
        
        for i in range(len(layers_shape) - 1):
            if no_act:
                layers += [nn.Linear(layers_shape[i], layers_shape[i+1])]
                act_func_name = "No Activation"
            else:
                layers += [nn.Linear(layers_shape[i], layers_shape[i+1]), act_func]
        
        layers += [nn.Linear(layers_shape[-1], num_labels)]
        self.layers = nn.Sequential(*layers)
        
        model_name = f"SimpleNetwork-{type(act_func).__name__}"
        self.config = {
            "model_name": model_name,
            "act_func": act_func_name,
            "input_shape": input_shape,
            "num_labels": num_labels,
            "hidden_shape": hidden_shape
        }

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)
