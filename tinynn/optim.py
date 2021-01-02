from tensor import Tensor
import numpy as np


class Optimizer:
    """
    base class for optimizers
    """

    def __init__(self, params):
        self.params = params


    def zero_grad(self):
        for p in self.params:
            p.grad.data *=0


class SGD(Optimizer):
    def __init__(self, params, lr=0.001):
        super().__init__(params)

        self.lr = lr

    def step(self, zero_grad = True):
        for p in self.params:
            p.data -= p.grad.data * self.lr

            if(zero_grad):
                p.grad.data *=0

class RMSprop(Optimizer):
    def __init__(self, params, lr=0.001, decay=0.88, eps=0.8):
        super().__init__(params)
        self.lr, self.decay, self.eps = lr, decay, eps
        #TODO add support for zeros and ones Tensors
        self.s = [Tensor(np.zeros(p.data.shape), requires_grad=False) for p in self.params]

    def step(self, zero_grad= True):
        
        for i , p in enumerate(self.params):
            self.s[i].data += self.s[i].data*self.decay + (1-self.decay)*np.square(p.grad.data)
            p.data -= self.lr*(1/(self.eps +np.sqrt(self.s[i].data)))*p.grad.data
            
            if(zero_grad):
                p.grad.data *=0



class Adam(Optimizer):
    def __init__(self, params, lr=0.001):
        super().__init__(params)
        raise NotImplementedError

    def step(self):
        raise NotImplementedError
