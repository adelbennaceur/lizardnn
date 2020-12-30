import numpy as np


class Tensor(object):
    def __init__(self, data, autograd=False, creators=None, creation_op=None, id=None):

        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None
        self.autograd = autograd
        self.children = {}

        if id is None:
            id = np.random.randint(0, 10000)
        self.id = id

        if creators is not None:
            for c in creators:
                # keep track of how many children a tensor has
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    def children_grads(self):
        # check whether the tensor received the grads of all of its children
        for id, it in self.children.items():
            if it != 0:
                return False
        return True

    def backward(self, grad, grad_origin=None):
        if self.autograd:
            if grad_origin is not None:
                # check if you can backpropagate then decrement the counter
                if self.children[grad_origin.id] == 0:
                    raise Exception("cannot backpropagate more than once")
                else:
                    self.children[grad_origin.id] -= 1

            # accumualte gradient from children
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            if self.creators is not None and (
                self.children_grads() or grad_origin is not None
            ):
                # setattr()
                if self.creation_op == "add":
                    # when two tensors are added together the resulthas two creators
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)

                if self.creation_op == "neg":
                    self.creators[0].backward(self.grad.__neg__())

                if self.creation_op == "mul":
                    new = self.grad.data * self.creators[1]
                    self.creators[0].backward(new, self)
                    new = self.grad.data * self.creators[0]
                    self.creators[1].backward(new, self)

                if self.creation_op == "sub":
                    new = Tensor(self.grad.data)
                    self.creators[0].backward(self.grad.data, self)
                    new = Tensor(self.grad.__neg__().data)
                    self.creators[1].backward(new, self)

    def __repr__(self):
        return str(self.data.__repr__())

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(
                self.data + other.data,
                autograd=True,
                creators=[self, other],
                creation_op="add",
            )
        return Tensor(self.data + other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(
                self.data * (-1),
                autograd=True,
                creators=[self],
                creation_op="neg",
            )
        return Tensor(self.data * (-1))

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(
                self.data * other.data,
                autograd=True,
                creators=[self, other],
                creation_op="mul",
            )

        return Tensor(self.data * other.data)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(
                self.data - other.data,
                autograd=True,
                creators=[self, other],
                creation_op="sub",
            )
        return Tensor(self.data - other.data)

    def sum(self, dim):
        # sum over a dimension
        if self.autograd:

            return Tensor(
                self.data.sum(dim), autograd=True, creators=[self], creation_op="sum"
            )
        return Tensor(self.data.sum(dim))


if __name__ == "__main__":
    a = Tensor([1, 2, 3, 4, 5], autograd=True)
    b = Tensor([0, 5, 4, 1, 8], autograd=True)
    c = Tensor([1, 1, 1, 1, 1], autograd=True)

    d = a + (-b)
    e = (-b) + c
    f = d + e

    f.backward(Tensor([1, 1, 1, 1, 1]))

    # f.backward(Tensor([1, 1, 1, 1, 1]))

    print(b.grad.data == np.array([-2, -2, -2, -2, -2]))

    print(d.grad.data)
    print(f.creators)
    print(f.creation_op)
