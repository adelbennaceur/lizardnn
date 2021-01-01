from tinynn.tensor import Tensor


class Optimizer:
    """
    base class for optimizers
    """

    def __init__(self, params):
        self.params = [p for p in params if p.requires_grad]

    def zero_grad(self):
        for param in self.params:
            param.grad.data *= 0


class SGD(Optimizer):
    def __init__(self, params, lr=0.001):
        super.__init__(params)

        self.lr = lr

    def step(self):
        for p in self.params:
            p -= p.grad.data * self.lr


class RMSprop(Optimizer):
    def __init__(self, params):
        super.__init__(params)
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self, params, lr=0.001):
        super.__init__(params)
        raise NotImplementedError

    def step(self):
        raise NotImplementedError
