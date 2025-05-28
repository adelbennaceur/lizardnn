# LizardNN 🦎

A minimal deep learning library built on NumPy for educational purposes. LizardNN implements a dynamic computational graph with automatic differentiation.

## Features

- **Automatic Differentiation Engine**
  - Dynamic computational graph construction
  - Backpropagation implementation
  - Support for basic mathematical operations

- **Neural Network Components**
  - Layers: Linear (fully connected), Sequential
  - Activations: ReLU, LeakyReLU, Tanh, Sigmoid
  - Regularization: Dropout
  - Weight Initialization: Xavier/Glorot, He/Kaiming

- **Optimizers**
  - SGD
  - RMSProp
  - Adam

- **Loss Functions**
  - MSE (Mean Squared Error)
  - Cross Entropy
  - Binary Cross Entropy
  - L1 Loss

## Installation

Make sure `uv` is installed.

```bash
uv venv
source .venv/bin/activate
uv pip install .
```

## Quick Start

### Basic Tensor Operations

```python
from lizardnn.tensor import Tensor

# create tensors with automatic gradient tracking
a = Tensor([1, 2, 3, 4, 5], requires_grad=True)
b = Tensor([2, 2, 2, 2, 2], requires_grad=True)

# perform operations
c = a + b
d = c.sigmoid()
e = d * c

# compute gradients
e.backward()

# Expected ? (compute it yourself)
print("Gradient of b:", b.grad.data)
print("Gradient of a:", a.grad.data)

```

### Creating a Neural Network

```python
import numpy as np

from lizardnn.layers import Dropout, Linear, ReLU, Sequential
from lizardnn.loss import MSELoss
from lizardnn.optim import Adam
from lizardnn.tensor import Tensor

# create a simple feed-forward neural network
model = Sequential(
    [
        Linear(2, 64),
        ReLU(),
        Dropout(0.2),
        Linear(64, 32),
        ReLU(),
        Linear(32, 1),
    ]
)

# prepare data
X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=True)
y = Tensor([[0], [1], [1], [0]], requires_grad=True)

# initialize loss and optimizer
criterion = MSELoss()
optimizer = Adam(parameters=model.get_parameters(), lr=0.01)

# training loop
for epoch in range(100):
    # Forward pass
    pred = model(X)
    loss = criterion(pred, y)

    # Backward pass
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        pred_label = (pred.data > 0.5).astype(np.float32)
        acc = (pred_label == y.data).mean() * 100
        print(f"Epoch {epoch}, Loss: {loss.data}, Acc: {acc:.2f}%")
```

## Project Structure

```
lizardnn/
├── __init__.py
├── tensor.py     # Core tensor operations and autograd
├── layers.py     # Neural network layers
├── loss.py       # Loss functions
├── optim.py      # Optimizers
└── examples/     # Usage examples
    ├── logistic_regression.py
    ├── mlp.py
    ├── example_classification.py
    └── example_regression.py
```

## License

MIT License - feel free to use this for learning and educational purposes.

## Acknowledgments

Originally developed in 2020 early in my career, this project was built to deepen my understanding of the internals of deep learning libraries. Inspired by modern frameworks like PyTorch, it was fully refactored in 2025 to align with best practices and cleaner architectural design.
