from tensor import Tensor
import numpy as np


class Layer:
    """
    base class for different neural network layers
    """

    def __init__(self, params):
        self.params = list()

    def get_parameters(self):
        return self.params


class Linear(Layer):
    """
    Apply linear transformation to input data , y = x*A.T + b
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()

        # TODO: implement random init function (mean ,std)
        #He initialization

        w = np.random.randn(in_features, out_features) * np.sqrt(
            2.0 / in_features + out_features
        )
        self.weight = Tensor(w, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        self.params.append(self.weight)
        self.params.append(self.bias)

    def forward(self, x):
        x = x.mm(self.weight) + self.bias.expand(0, len(x.data))
        return x


class Sequential(Layer):
    def __init__(self, layers=list()):
        super().__init__()
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def get_parameters(self):
        params = list()

        for layer in self.layers:
            params.append(layer.get_parameters())
        return params

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


class Conv2d(Layer):
    """
    2d convolution over an input signal containing diffeent input planes
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        strdie,
        padding,
        dilation,
        bias=True,
    ):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
