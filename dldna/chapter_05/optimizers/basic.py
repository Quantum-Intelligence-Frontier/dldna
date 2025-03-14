from typing import Iterable, List, Optional
import torch
from torch import nn

from dldna.chapter_05.optimizers.base import BaseOptimizer

class SGD(BaseOptimizer):
    """Implements Stochastic Gradient Descent (SGD) with optional momentum and weight decay.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate.
        maximize: Whether to maximize or minimize the objective (default: False).
        momentum: Momentum factor (default: 0.0).
        weight_decay: Weight decay (L2 penalty) (default: 0.0).
    """
    def __init__(self, params: Iterable[nn.Parameter], lr: float,
                 maximize: bool = False, momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.maximize = maximize
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.momentum_buffer_list: List[Optional[torch.Tensor]] = [None] * len(self.params)

    @torch.no_grad()
    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad if not self.maximize else -p.grad

            # Weight decay (applied before momentum)
            if self.weight_decay != 0:
                grad = grad.add(p, alpha=self.weight_decay)


            if self.momentum != 0.0:
                buf = self.momentum_buffer_list[i]
                if buf is None:
                    buf = torch.clone(grad).detach()
                else:
                    buf.mul_(self.momentum).add_(grad, alpha=1 - self.momentum)
                grad = buf
                self.momentum_buffer_list[i] = buf

            p.add_(grad, alpha=-self.lr)

class Adam(BaseOptimizer):
    """Implements the Adam optimizer.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate.
        beta1: Exponential decay rate for the first moment estimates (default: 0.9).
        beta2: Exponential decay rate for the second moment estimates (default: 0.999).
        eps: Term added to the denominator to improve numerical stability (default: 1e-8).
    """
    def __init__(self, params: Iterable[nn.Parameter], lr: float,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.state_steps = {p: 0 for p in self.params}
        self.exp_avgs_m = {p: torch.zeros_like(p.data) for p in self.params}
        self.exp_avgs_v = {p: torch.zeros_like(p.data) for p in self.params}

    @torch.no_grad()
    def step(self) -> None:
        for p in self.params:
            if p.grad is None: #Skip parameters without gradients
                continue

            self.state_steps[p] += 1

            self.exp_avgs_m[p].mul_(self.beta1).add_(p.grad, alpha=1 - self.beta1)
            self.exp_avgs_v[p].mul_(self.beta2).addcmul_(p.grad, p.grad, value=1 - self.beta2)

            bias_correction1 = 1 - self.beta1 ** self.state_steps[p]
            bias_correction2 = 1 - self.beta2 ** self.state_steps[p]

            m_hat = self.exp_avgs_m[p] / bias_correction1
            v_hat = self.exp_avgs_v[p] / bias_correction2

            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

class AdaGrad(BaseOptimizer):
    """Implements the AdaGrad optimizer.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 1e-2).
        eps: Term added to the denominator to improve numerical stability (default: 1e-10).
    """
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 1e-2,
                 eps: float = 1e-10):
        super().__init__(params, lr)
        self.eps = eps
        self.square_avgs = {p: torch.zeros_like(p) for p in self.params}

    @torch.no_grad()
    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad
            square_avg = self.square_avgs[p]
            square_avg.addcmul_(grad, grad, value=1)
            avg = square_avg.sqrt().add_(self.eps)
            p.addcdiv_(grad, avg, value=-self.lr)


import torch
import torch.nn as nn
from typing import Iterable, Optional, List

class AdamW(object):  # BaseOptimizer 상속 생략 (단순화 목적)
    """Implements the AdamW optimizer.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate.
        beta1: Exponential decay rate for the first moment estimates (default: 0.9).
        beta2: Exponential decay rate for the second moment estimates (default: 0.999).
        eps: Term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay: Weight decay (L2 penalty) (default: 0.0).
        maximize: Whether to maximize or minimize the objective (default: False)
    """
    def __init__(self, params: Iterable[nn.Parameter], lr: float,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 0.0, maximize: bool = False):
        #super().__init__(params, lr) # BaseOptimizer 상속 생략
        self.params = list(params)  # Ensure params is a list
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.maximize = maximize # 추가

        self.state_steps = {p: 0 for p in self.params}
        self.exp_avgs_m = {p: torch.zeros_like(p.data) for p in self.params}
        self.exp_avgs_v = {p: torch.zeros_like(p.data) for p in self.params}

    @torch.no_grad()
    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad if not self.maximize else -p.grad  # maximize 추가

            self.state_steps[p] += 1

            # Decoupled weight decay (applied before momentum updates)
            if self.weight_decay != 0:
                p.data.mul_(1 - self.lr * self.weight_decay)

            self.exp_avgs_m[p].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            self.exp_avgs_v[p].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            bias_correction1 = 1 - self.beta1 ** self.state_steps[p]
            bias_correction2 = 1 - self.beta2 ** self.state_steps[p]

            m_hat = self.exp_avgs_m[p] / bias_correction1
            v_hat = self.exp_avgs_v[p] / bias_correction2

            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)