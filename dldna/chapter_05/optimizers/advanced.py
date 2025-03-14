from typing import Iterable
import torch
from torch import nn
from dldna.chapter_05.optimizers.base import BaseOptimizer

class Lion(BaseOptimizer):
    """The Lion optimizer from Google Brain.

    Implements the Lion (Evolved Sign Momentum) optimizer.
    """
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 1e-4,
                 betas: tuple = (0.9, 0.99), weight_decay: float = 0.0):
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter: {betas}')

        super().__init__(params, lr)
        self.betas = betas
        self.weight_decay = weight_decay
        self.momentum = {p: torch.zeros_like(p) for p in self.params}

    @torch.no_grad()
    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad
            if self.weight_decay != 0:
                grad = grad.add(p, alpha=self.weight_decay)

            mom = self.momentum[p]
            update = mom.mul(self.betas[0]).add(grad, alpha=1 - self.betas[0])
            p.add_(update.sign(), alpha=-self.lr)
            mom.mul_(self.betas[1]).add_(grad, alpha=1 - self.betas[1])

class Sophia(BaseOptimizer):
    """The Sophia optimizer.

    Implements the Sophia (Second-order Clipped Stochastic Optimization) optimizer.
    """
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 1e-3,
                 betas: tuple = (0.965, 0.99), rho: float = 0.04,
                 weight_decay: float = 0.0, k: int = 10):
        super().__init__(params, lr)
        self.betas = betas
        self.rho = rho
        self.weight_decay = weight_decay
        self.k = k

        self.exp_avg = {p: torch.zeros_like(p) for p in self.params}
        self.hessian = {p: torch.ones_like(p) for p in self.params}
        self.steps = 0

    @torch.no_grad()
    def step(self) -> None:
        self.steps += 1
        update_hessian = (self.steps % self.k == 0)

        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad
            if self.weight_decay != 0:
                grad = grad.add(p, alpha=self.weight_decay)

            state = self.exp_avg[p]
            state.mul_(self.betas[0]).add_(grad, alpha=1 - self.betas[0])

            if update_hessian:
                self.hessian[p].mul_(self.betas[1]).addcmul_(
                    grad, grad, value=1 - self.betas[1]
                )

            denom = self.rho * self.hessian[p] + 1e-8  # Added epsilon for numerical stability.
            update = state.div(denom).clamp(min=-1.0, max=1.0) #Clamping
            p.add_(update, alpha=-self.lr)