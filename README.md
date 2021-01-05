# lizardnn
lizardnn is a minimal deep learning library written in python and built on top of NumPy, the library for learning purposes only.

The library has a built-in automatic differentiation engine that implements backpropagation and dynamically builds a computational graph, the library is also built with basic features to train a simple neural network such as basic operations, optimizers, loss functions, and layers.


### Features


- A simple autograd engine (dynamically built computational graph)
- Layers: Linear, Sequential.
- Optimizers: SGD, RMSProp, Adam
- Loss: MSELoss, CrossEntropyLoss)



### basic tensor operations example

```

from lizardnn.tensor import Tensor

a = Tensor([1,2,3,4,5], requires_grad=True)
b = Tensor([2,2,2,2,2], requires_grad=True)
c = Tensor([5,4,3,2,1], requires_grad=True)

d = a + b
e = d.sigmoid()
f = d * e

f.backward()

print(b.grad.data)
print(a.grad.data)

```


### MLP example 
```

from lizardnn.tensor import Tensor
from lizardnn.layers import Linear, Linear, Sequential
from lizardnn.optim import SGD
from lizardnn.loss import MSEloss


data = Tensor([[0,0],[0,1],[1,0],[1,1]], requires_grad=True)
    target = Tensor([[0],[1],[0],[1]], requires_grad=True)

    model = Sequential([Linear(2,3), Linear(3,1)])

    criterion = MSEloss()
    
  
    optim = SGD(parameters=model.get_parameters(), lr=0.01)

    for i in range(10):

        pred = model(data)
    
        loss = criterion(pred, target)

        loss.backward()
        optim.step()
        print(loss)