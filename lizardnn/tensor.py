from typing import List, Optional, Union, Tuple, Dict
import numpy as np
from numpy import ndarray


class Tensor:
    """Core tensor class implementing automatic differentiation.

    This class wraps numpy arrays and tracks operations for automatic
    differentiation (autograd). It builds a dynamic computational graph
    and implements the backward pass using reverse-mode differentiation.
    """

    def __init__(
        self,
        data: Union[float, int, list, ndarray],
        requires_grad: bool = False,
        creators: Optional[List["Tensor"]] = None,
        creation_op: Optional[str] = None,
        id: Optional[int] = None,
    ) -> None:
        """Initialize a new tensor.

        Args:
            data: Input data (will be converted to numpy array)
            requires_grad: If True, gradients will be computed
            creators: List of tensors that created this tensor
            creation_op: The operation that created this tensor
            id: Unique identifier for the tensor
        """
        self.data = np.array(data, dtype=np.float32)
        self.creation_op = creation_op
        self.creators = creators
        self.grad: Optional[Tensor] = None
        self.requires_grad = requires_grad
        self.children: Dict[int, int] = {}

        if id is None:
            id = np.random.randint(0, 10000)
        self.id = id

        if creators is not None:
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    def children_grads(self) -> bool:
        """Check if all children have propagated their gradients."""
        return all(it == 0 for it in self.children.values())

    def backward(
        self, grad: Optional["Tensor"] = None, grad_origin: Optional["Tensor"] = None
    ) -> None:
        """Compute gradients through backpropagation.

        Args:
            grad: Gradient from the next layer
            grad_origin: Tensor that called backward on this tensor

        Raises:
            Exception: If trying to backpropagate more than once through a path
        """
        if not self.requires_grad:
            return

        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        if grad_origin is not None:
            if self.children[grad_origin.id] == 0:
                raise Exception("Cannot backpropagate more than once")
            self.children[grad_origin.id] -= 1

        # Accumulate gradients
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        assert not grad.requires_grad, "Gradients should not require gradients"

        # Backpropagate to creators if all children have propagated their gradients
        if self.creators is not None and (self.children_grads() or grad_origin is None):
            if self.creation_op == "add":
                self.creators[0].backward(self.grad, self)
                self.creators[1].backward(self.grad, self)

            elif self.creation_op == "neg":
                self.creators[0].backward(self.grad.__neg__(), self)

            elif self.creation_op == "mul":
                g0 = Tensor(self.grad.data * self.creators[1].data)
                self.creators[0].backward(g0, self)
                g1 = Tensor(self.grad.data * self.creators[0].data)
                self.creators[1].backward(g1, self)

            elif self.creation_op == "sub":
                new = Tensor(self.grad.data)
                self.creators[0].backward(new, self)
                new = Tensor(self.grad.__neg__().data)
                self.creators[1].backward(new, self)

            elif self.creation_op == "transpose":
                self.creators[0].backward(self.grad.transpose(), self)

            elif "pow" in self.creation_op:
                p = float(self.creation_op.split("_")[1])
                self.creators[0].backward(
                    Tensor((p * self.creators[0].data ** (p - 1)) * self.grad.data),
                    self,
                )

            elif self.creation_op == "mm":
                act = self.creators[0]
                weights = self.creators[1]
                new = self.grad.mm(weights.transpose())
                act.backward(new, self)
                new = self.grad.transpose().mm(act).transpose()
                weights.backward(new, self)

            elif "sum" in self.creation_op:
                dim = int(self.creation_op.split("_")[1])
                ds = self.creators[0].data.shape[dim]
                self.creators[0].backward(self.grad.expand(dim, ds))

            elif "expand" in self.creation_op:
                dim = int(self.creation_op.split("_")[1])
                self.creators[0].backward(self.grad.sum(dim))

            elif self.creation_op == "sigmoid":
                self.creators[0].backward(
                    Tensor(self.grad.data * (self.data * (1 - self.data))), self
                )

            elif self.creation_op == "tanh":
                self.creators[0].backward(
                    Tensor(self.grad.data * (1 - self.data * self.data)), self
                )

            elif self.creation_op == "relu":
                self.creators[0].backward(
                    Tensor(self.grad.data * (self.data > 0), requires_grad=False), self
                )

            elif self.creation_op == "leaky_relu":
                mask = self.data > 0
                grad_data = np.where(mask, 1.0, self.negative_slope)
                self.creators[0].backward(
                    Tensor(self.grad.data * grad_data, requires_grad=False), self
                )

            elif self.creation_op == "cross_entropy":
                n = self.target_dist.shape[0]
                dx = (self.softmax_out - self.target_dist) / n
                self.creators[0].backward(Tensor(dx), self)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other: Union["Tensor", float, int]) -> "Tensor":
        """Add two tensors."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.requires_grad or other.requires_grad:
            return Tensor(
                self.data + other.data,
                requires_grad=True,
                creators=[self, other],
                creation_op="add",
            )
        return Tensor(self.data + other.data)

    def __neg__(self) -> "Tensor":
        """Negate tensor."""
        if self.requires_grad:
            return Tensor(
                -self.data,
                requires_grad=True,
                creators=[self],
                creation_op="neg",
            )
        return Tensor(-self.data)

    def __mul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        """Multiply two tensors element-wise."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.requires_grad or other.requires_grad:
            return Tensor(
                self.data * other.data,
                requires_grad=True,
                creators=[self, other],
                creation_op="mul",
            )
        return Tensor(self.data * other.data)

    def __sub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        """Subtract two tensors."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.requires_grad or other.requires_grad:
            return Tensor(
                self.data - other.data,
                requires_grad=True,
                creators=[self, other],
                creation_op="sub",
            )
        return Tensor(self.data - other.data)

    def __truediv__(self, other: Union["Tensor", float, int]) -> "Tensor":
        """Divide two tensors element-wise."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other**-1)

    def __pow__(self, exp: float) -> "Tensor":
        """Raise tensor to a power."""
        if self.requires_grad:
            return Tensor(
                self.data**exp,
                requires_grad=True,
                creators=[self],
                creation_op=f"pow_{exp}",
            )
        return Tensor(self.data**exp)

    def mm(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication."""
        if self.requires_grad or other.requires_grad:
            return Tensor(
                self.data.dot(other.data),
                requires_grad=True,
                creators=[self, other],
                creation_op="mm",
            )
        return Tensor(self.data.dot(other.data))

    def sum(self, dim: int) -> "Tensor":
        """Sum tensor along dimension."""
        if self.requires_grad:
            return Tensor(
                self.data.sum(dim),
                requires_grad=True,
                creators=[self],
                creation_op=f"sum_{dim}",
            )
        return Tensor(self.data.sum(dim))

    def mean(self, dim: Optional[int] = None) -> "Tensor":
        """Mean of tensor along dimension."""
        if dim is None:
            return Tensor(self.data.mean())
        return self.sum(dim) / self.data.shape[dim]

    def transpose(self) -> "Tensor":
        """Transpose tensor."""
        if self.requires_grad:
            return Tensor(
                self.data.transpose(),
                requires_grad=True,
                creators=[self],
                creation_op="transpose",
            )
        return Tensor(self.data.transpose())

    def expand(self, dim: int, copies: int) -> "Tensor":
        """Expand tensor along dimension."""
        trans = list(range(len(self.data.shape)))
        trans.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape).transpose(trans)

        if self.requires_grad:
            return Tensor(
                new_data,
                requires_grad=True,
                creators=[self],
                creation_op=f"expand_{dim}",
            )
        return Tensor(new_data)

    def sigmoid(self) -> "Tensor":
        """Apply sigmoid activation."""
        if self.requires_grad:
            return Tensor(
                1.0 / (1.0 + np.exp(-self.data)),
                requires_grad=True,
                creators=[self],
                creation_op="sigmoid",
            )
        return Tensor(1.0 / (1.0 + np.exp(-self.data)))

    def tanh(self) -> "Tensor":
        """Apply tanh activation."""
        if self.requires_grad:
            return Tensor(
                np.tanh(self.data),
                requires_grad=True,
                creators=[self],
                creation_op="tanh",
            )
        return Tensor(np.tanh(self.data))

    def relu(self) -> "Tensor":
        """Apply ReLU activation."""
        if self.requires_grad:
            return Tensor(
                np.maximum(0, self.data),
                requires_grad=True,
                creators=[self],
                creation_op="relu",
            )
        return Tensor(np.maximum(0, self.data))

    def leaky_relu(self, negative_slope: float = 0.01) -> "Tensor":
        """Apply Leaky ReLU activation."""
        if self.requires_grad:
            out = Tensor(
                np.where(self.data > 0, self.data, self.data * negative_slope),
                requires_grad=True,
                creators=[self],
                creation_op="leaky_relu",
            )
            out.negative_slope = negative_slope
            return out
        return Tensor(np.where(self.data > 0, self.data, self.data * negative_slope))

    def cross_entropy(self, target_idx: "Tensor") -> "Tensor":
        """Compute cross entropy loss.

        Args:
            target_idx: Target indices

        Returns:
            Loss tensor
        """
        # Compute softmax
        exp_data = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        softmax_out = exp_data / exp_data.sum(axis=-1, keepdims=True)

        # Compute cross entropy
        t = target_idx.data.flatten()
        p = softmax_out.reshape(len(t), -1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p + 1e-7) * target_dist).sum(1).mean()

        if self.requires_grad:
            out = Tensor(
                loss,
                requires_grad=True,
                creators=[self],
                creation_op="cross_entropy",
            )
            out.softmax_out = softmax_out
            out.target_dist = target_dist
            return out

        return Tensor(loss)
