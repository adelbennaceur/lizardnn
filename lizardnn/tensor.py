import numpy as np


class Tensor(object):
    def __init__(
        self, data, requires_grad=False, creators=None, creation_op=None, id=None
    ):

        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None
        self.requires_grad = requires_grad
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

    def backward(self, grad=None, grad_origin=None):

        if self.requires_grad:

            if grad is None:
                grad = Tensor(np.ones_like(self.data))

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

            # grads must not have grads of their own
            assert grad.requires_grad == False

            if self.creators is not None and (
                self.children_grads() or grad_origin is None
            ):
                # setattr()
                if self.creation_op == "add":
                    # when two tensors are added together the resulthas two creators
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)

                if self.creation_op == "neg":
                    self.creators[0].backward(self.grad.__neg__())

                if self.creation_op == "mul":
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new, self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new, self)

                if self.creation_op == "sub":
                    new = Tensor(self.grad.data)
                    self.creators[0].backward(new, self)
                    new = Tensor(self.grad.__neg__().data)
                    self.creators[1].backward(new, self)

                if self.creation_op == "transpose":
                    self.creators[0].backward(self.grad.transpose())

                if "pow" in self.creation_op:
                    p = int(self.creation_op.split("_")[1])

                    self.creators[0].backward(self.grad.pow(p))

                # matrix multiplication
                if self.creation_op == "mm":
                    # previous layer activation
                    activation = self.creators[0]
                    weights = self.creators[1]
                    new = self.grad.mm(weights.transpose())
                    activation.backward(new)
                    new = self.grad.transpose().mm(activation).transpose()
                    weights.backward(new)

                if "sum" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    ds = self.creators[0].data.shape[dim]
                    self.creators[0].backward(self.grad.expand(dim, ds))

                if "expand" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))

    def __repr__(self):
        return str(self.data.__repr__())

    def __add__(self, other):
        if self.requires_grad and other.requires_grad:
            return Tensor(
                self.data + other.data,
                requires_grad=True,
                creators=[self, other],
                creation_op="add",
            )
        return Tensor(self.data + other.data)

    def __neg__(self):
        if self.requires_grad:
            return Tensor(
                self.data * (-1),
                requires_grad=True,
                creators=[self],
                creation_op="neg",
            )
        return Tensor(self.data * (-1))

    def __mul__(self, other):
        if self.requires_grad and other.requires_grad:
            return Tensor(
                self.data * other.data,
                requires_grad=True,
                creators=[self, other],
                creation_op="mul",
            )

        return Tensor(self.data * other.data)

    def __sub__(self, other):
        if self.requires_grad and other.requires_grad:
            return Tensor(
                self.data - other.data,
                requires_grad=True,
                creators=[self, other],
                creation_op="sub",
            )
        return Tensor(self.data - other.data)

    def pow(self, exp):
        if self.requires_grad:
            return Tensor(
                np.power(self.data, exp),
                requires_grad=True,
                creators=[self],
                creation_op="pow_" + str(exp),
            )

        return Tensor(np.power(self.data, exp))

    def mm(self, other):
        if self.requires_grad:
            return Tensor(
                self.data.dot(other.data),
                requires_grad=True,
                creators=[self, other],
                creation_op="mm",
            )

        return Tensor(self.data.dot(other.data))

    def sum(self, dim):
        # sum over a dimension
        if self.requires_grad:

            return Tensor(
                self.data.sum(dim),
                requires_grad=True,
                creators=[self],
                creation_op="sum_" + str(dim),
            )
        return Tensor(self.data.sum(dim))

    def transpose(self):
        if self.requires_grad:
            return Tensor(
                self.data.transpose(),
                requires_grad=True,
                creators=[self],
                creation_op="transpose",
            )

        return Tensor(self.data.transpose())

    def expand(self, dim, copies):
        trans = list(range(0, len(self.data.shape)))
        trans.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape).transpose(trans)

        if self.requires_grad:
            return Tensor(
                new_data,
                requires_grad=True,
                creators=[self],
                creation_op="expand_" + str(dim),
            )

        return Tensor(new_data)
