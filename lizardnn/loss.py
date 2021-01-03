from tensor import Tensor


class MSEloss(object):
    """
    Criterion that measures the mean squared error
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self, preds, target):
        return (preds - target).pow(2).sum(0)


class CrossEntropyLosss(object):
    """
    cross entropy loss criterion
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, target):
        return preds.cross_entropy(target)
