from lizardnn.tensor import Tensor
import numpy as np



#TODO : implement different  weight initialization techniques


def init_weights(out_features, in_features, weight_init):
    pass


class Layer:
    """
    base class for different neural network layers
    """
    
    def __init__(self):
        self.params = list()

    def get_parameters(self):
        return self.params

    def forward(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        return self.forward(*args)


class Linear(Layer):
    """
    Apply linear transformation to input data , y = x*A.T + b
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()

        # TODO: implement random init function (mean ,std)
        # He initialization

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

    def __iter__(self):
        yield from self.layers

    def __getitem__(self, idx):
        return self.layers[idx]

    def add(self, layer):
        self.layers.append(layer)

    def get_parameters(self):
        params = list()

        for layer in self.layers:
            params += layer.get_parameters()
        return params

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

#-----------activations----------
class Tanh(Layer):
    """
    Tanh activation layer
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.tanh()


class Sigmoid(Layer):
    """
    Tanh activation layer
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.sigmoid()

