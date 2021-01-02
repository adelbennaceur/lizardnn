#!/usr/bin/env python

from tensor import Tensor
from optim import SGD, RMSprop
import numpy as np


def main():

    data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), requires_grad=True)
    gts = Tensor(np.array([[0], [0], [0], [1]]), requires_grad=True)

    weights = list()

    w1 = Tensor(np.random.rand(2, 3), requires_grad=True)
    w2 = Tensor(np.random.rand(3, 1), requires_grad=True)

    weights.append(w1)
    weights.append(w2)

    lr = 0.01
    optimizer = RMSprop(params=weights, lr=lr)

    for i in range(100):

        pred = data.mm(w1).mm(w2)

        loss = ((pred - gts) * (pred - gts)).sum(0)

        loss.backward()
        optimizer.step()

        # without optimizer
        # for w in weights:
        #    w.data -= w.grad.data * lr
        #    w.grad.data *= 0

        print("epoch: ", i, "--------------   Loss", loss.data)


if __name__ == "__main__":
    main()
