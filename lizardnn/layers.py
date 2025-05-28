from enum import Enum
from typing import Any, Iterator, List, Optional, Union

import numpy as np  # type: ignore

from lizardnn.tensor import Tensor


class WeightInit(Enum):
    """Weight initialization methods"""

    XAVIER = "xavier"
    HE = "he"
    STANDARD = "standard"


def init_weights(
    out_features: int, in_features: int, weight_init: WeightInit = WeightInit.HE
) -> np.ndarray:
    """Initialize weights for neural network layers.

    Args:
        out_features: Number of output features
        in_features: Number of input features
        weight_init: Initialization method to use

    Returns:
        numpy.ndarray: Initialized weights
    """
    if weight_init == WeightInit.XAVIER:
        std = np.sqrt(2.0 / (in_features + out_features))
        return np.random.randn(in_features, out_features) * std
    elif weight_init == WeightInit.HE:
        std = np.sqrt(2.0 / in_features)
        return np.random.randn(in_features, out_features) * std
    else:
        return np.random.randn(in_features, out_features) * 0.01


class Layer:
    """Base class for neural network layers.

    All layer implementations should inherit from this class
    and implement the forward method.
    """

    def __init__(self) -> None:
        """Initialize the layer with empty parameter list."""
        self.params: List[Tensor] = []

    def get_parameters(self) -> List[Tensor]:
        """Get all trainable parameters of the layer.

        Returns:
            List of parameter tensors
        """
        return self.params

    def forward(self, *args: Any) -> Tensor:
        """Forward pass of the layer.

        Args:
            *args: Input tensors

        Returns:
            Output tensor

        Raises:
            NotImplementedError: If not implemented by child class
        """
        raise NotImplementedError

    def __call__(self, *args: Any) -> Tensor:
        """Make layer callable, equivalent to forward pass."""
        return self.forward(*args)


class Linear(Layer):
    """Linear (fully connected) layer implementing y = xW + b

    Applies a linear transformation to the incoming data.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init: WeightInit = WeightInit.HE,
    ) -> None:
        """Initialize the linear layer.

        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If set to False, layer will not learn an additive bias
            weight_init: Weight initialization method to use
        """
        super().__init__()

        w = init_weights(out_features, in_features, weight_init)
        self.weight = Tensor(w, requires_grad=True)
        self.params.append(self.weight)

        if bias:
            self.bias: Union[Tensor, None] = Tensor(
                np.zeros(out_features), requires_grad=True
            )
            self.params.append(self.bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of linear layer.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        out = x.mm(self.weight)
        if self.bias is not None:
            out = out + self.bias.expand(0, len(x.data))
        return out


class Sequential(Layer):
    """A sequential container of layers.

    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can be passed in.
    """

    def __init__(self, layers: Optional[List[Layer]] = None) -> None:
        """Initialize Sequential container.

        Args:
            layers: Optional list of layers to add
        """
        super().__init__()
        self.layers = layers or []

    def __iter__(self) -> Iterator[Layer]:
        """Iterate over contained layers."""
        yield from self.layers

    def __getitem__(self, idx: int) -> Layer:
        """Get layer by index."""
        return self.layers[idx]

    def add(self, layer: Layer) -> None:
        """Add a new layer to the container.

        Args:
            layer: Layer to add
        """
        self.layers.append(layer)

    def get_parameters(self) -> List[Tensor]:
        """Get parameters from all layers.

        Returns:
            List of all parameters from all layers
        """
        params = []
        for layer in self.layers:
            params.extend(layer.get_parameters())
        return params

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all layers in sequence.

        Args:
            x: Input tensor

        Returns:
            Output tensor after passing through all layers
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x


class ReLU(Layer):
    """Rectified Linear Unit (ReLU) activation function."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU activation: max(0, x)"""
        return x.relu()


class LeakyReLU(Layer):
    """Leaky ReLU activation function."""

    def __init__(self, negative_slope: float = 0.01) -> None:
        """Initialize Leaky ReLU.

        Args:
            negative_slope: Controls the angle of the negative slope
        """
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        """Apply Leaky ReLU activation"""
        return x.leaky_relu(self.negative_slope)


class Tanh(Layer):
    """Hyperbolic tangent (tanh) activation function."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Apply tanh activation"""
        return x.tanh()


class Sigmoid(Layer):
    """Sigmoid activation function."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Apply sigmoid activation"""
        return x.sigmoid()


class Dropout(Layer):
    """Dropout layer.

    Randomly zeroes some of the elements of the input tensor with probability p.
    """

    def __init__(self, p: float = 0.5) -> None:
        """Initialize dropout layer.

        Args:
            p: Probability of an element to be zeroed
        """
        super().__init__()
        self.p = p
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        """Apply dropout during training."""
        if not self.training or self.p == 0:
            return x

        mask = np.random.binomial(1, 1 - self.p, x.data.shape)
        return x * Tensor(mask, requires_grad=False) / (1 - self.p)
