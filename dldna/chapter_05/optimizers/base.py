import torch
from torch import nn
from typing import Iterable

class BaseOptimizer:
    """Base class for optimization algorithms."""
    def __init__(self, params: Iterable[nn.Parameter], lr: float):
        """Initializes the BaseOptimizer.

        Args:
            params: An iterable of parameters to optimize.
            lr: The learning rate.
        """
        self.params = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        """Sets gradients of all parameters to zero."""
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_().zero_()

    def step(self) -> None:
        """Performs a single optimization step (parameter update).

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError