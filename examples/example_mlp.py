import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("lizardnn"), "..")))


import numpy as np
from lizardnn.tensor import Tensor
from lizardnn.layers import Layer, Linear, Sigmoid, Tanh, Sequential
from lizardnn.optim import SGD
from lizardnn.loss import MSEloss

class LizardNet(Layer):
    def __init__(self):
        super().__init__()

        self.fc1 = Linear(2, 3)
        self.act1 = Tanh()
        self.fc2 = Linear(3, 1)
        self.act2 = Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


def main():

    data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), requires_grad=True)
    target = Tensor(np.array([[0], [1], [0], [1]]), requires_grad=True)

    model = LizardNet()

    criterion = MSEloss()

    optim = SGD(parameters=model.get_parameters(), lr=1)

    for i in range(10):

        pred = model(data)

        loss = criterion(pred, target)

        loss.backward()
        optim.step()

    print(model.get_parameters())


if __name__ == "__main__":
    main()
