from typing import List, Optional

import numpy as np

from lizardnn.tensor import Tensor


class Optimizer:
    """Base class for all optimizers.

    All optimizers should inherit from this class and implement
    the step method.
    """

    def __init__(self, parameters: List[Tensor]) -> None:
        """Initialize optimizer.

        Args:
            parameters: List of parameters to optimize
        """
        self.parameters = parameters

    def zero_grad(self) -> None:
        """Zero out gradients of all parameters."""
        for p in self.parameters:
            p.grad = None

    def step(self, zero_grad: bool = True) -> None:
        """Perform a single optimization step.

        Args:
            zero_grad: Whether to zero gradients after step

        Raises:
            NotImplementedError: If not implemented by child class
        """
        raise NotImplementedError


class SGD(Optimizer):
    """Implements stochastic gradient descent with momentum.

    Nesterov momentum is supported.
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.001,
        momentum: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        """Initialize SGD optimizer.

        Args:
            parameters: Parameters to optimize
            lr: Learning rate
            momentum: Momentum factor
            nesterov: Whether to use Nesterov momentum
        """
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov

        if momentum > 0:
            self.velocity = [
                Tensor(np.zeros_like(p.data), requires_grad=False) for p in parameters
            ]

    def step(self, zero_grad: bool = True) -> None:
        """Perform a single optimization step."""
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            if self.momentum > 0:
                # Update velocity
                self.velocity[i].data = (
                    self.momentum * self.velocity[i].data + p.grad.data
                )

                if self.nesterov:
                    # nesterov update
                    p.data -= self.lr * (
                        self.momentum * self.velocity[i].data + p.grad.data
                    )
                else:
                    # regular momentum update
                    p.data -= self.lr * self.velocity[i].data
            else:
                # regular SGD update
                p.data -= self.lr * p.grad.data

        if zero_grad:
            self.zero_grad()


class RMSprop(Optimizer):
    """Implements RMSprop algorithm.

    It is recommended to leave the parameters beta and epsilon to their default values.
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.001,
        beta: float = 0.9,
        eps: float = 1e-8,
    ) -> None:
        """Initialize RMSprop optimizer.

        Args:
            parameters: Parameters to optimize
            lr: Learning rate
            beta: Decay rate for moving average
            eps: Term added for numerical stability
        """
        super().__init__(parameters)
        self.lr = lr
        self.beta = beta
        self.eps = eps

        self.square_avg = [
            Tensor(np.zeros_like(p.data), requires_grad=False) for p in parameters
        ]

    def step(self, zero_grad: bool = True) -> None:
        """Perform a single optimization step."""
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            # update running average of squared gradients
            self.square_avg[i].data = self.beta * self.square_avg[i].data + (
                1 - self.beta
            ) * np.square(p.grad.data)

            # update parameters
            p.data -= (
                self.lr * p.grad.data / (np.sqrt(self.square_avg[i].data) + self.eps)
            )

        if zero_grad:
            self.zero_grad()


class Adam(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in 'Adam: A Method for Stochastic Optimization'.
    The implementation incorporates AMSGrad variant.
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
    ) -> None:
        """Initialize Adam optimizer.

        Args:
            parameters: Parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages
            eps: Term added for numerical stability
            weight_decay: Weight decay (L2 penalty)
            amsgrad: Whether to use the AMSGrad variant
        """
        super().__init__(parameters)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

        self.step_count = 0

        # Initialize momentum buffers
        self.exp_avg = [
            Tensor(np.zeros_like(p.data), requires_grad=False) for p in parameters
        ]
        self.exp_avg_sq = [
            Tensor(np.zeros_like(p.data), requires_grad=False) for p in parameters
        ]

        if amsgrad:
            self.max_exp_avg_sq = [
                Tensor(np.zeros_like(p.data), requires_grad=False) for p in parameters
            ]

    def step(self, zero_grad: bool = True) -> None:
        """Perform a single optimization step."""
        self.step_count += 1

        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            grad = p.grad.data
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data

            # update biased first moment estimate
            self.exp_avg[i].data = (
                self.betas[0] * self.exp_avg[i].data + (1 - self.betas[0]) * grad
            )

            # update biased second raw moment estimate
            self.exp_avg_sq[i].data = self.betas[1] * self.exp_avg_sq[i].data + (
                1 - self.betas[1]
            ) * np.square(grad)

            if self.amsgrad:
                # maintain the maximum of all 2nd moment running avg. till now
                np.maximum(
                    self.max_exp_avg_sq[i].data,
                    self.exp_avg_sq[i].data,
                    out=self.max_exp_avg_sq[i].data,
                )
                # use the max. for normalizing running avg. of grad
                denom = np.sqrt(self.max_exp_avg_sq[i].data) + self.eps
            else:
                denom = np.sqrt(self.exp_avg_sq[i].data) + self.eps

            # bias correction
            bias_correction1 = 1 - self.betas[0] ** self.step_count
            bias_correction2 = 1 - self.betas[1] ** self.step_count
            step_size = self.lr * np.sqrt(bias_correction2) / bias_correction1

            # update params
            p.data -= step_size * self.exp_avg[i].data / denom

        if zero_grad:
            self.zero_grad()
