import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("lizardnn"), "..")))


import numpy as np
from lizardnn.tensor import Tensor
from lizardnn.layers import Linear, Linear, Sequential
from lizardnn.optim import SGD
from lizardnn.loss import MSEloss


def main():

    data = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=True)
    target = Tensor([[0], [1], [0], [1]], requires_grad=True)

    model = Sequential([Linear(2, 3), Linear(3, 1)])

    criterion = MSEloss()

    optim = SGD(parameters=model.get_parameters(), lr=0.01)

    for i in range(10):

        pred = model(data)

        loss = criterion(pred, target)

        loss.backward()
        optim.step()
        print(loss)


if __name__ == "__main__":
    #main()
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
