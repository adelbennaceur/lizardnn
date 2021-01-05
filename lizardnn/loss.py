from lizardnn.tensor import Tensor


class MSEloss(object):
    """
    Criterion that measures the mean squared error
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, target):
        return ((preds - target) * (preds - target)).sum(0)

    def __call__(self, *args):
        return self.forward(*args)


class CrossEntropyLoss(object):
    """
    cross entropy loss criterion
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, target):
        return preds.cross_entropy(target)

    def __call__(self, *args):
        return self.forward(*args)
