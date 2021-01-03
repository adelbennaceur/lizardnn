from tensor import Tensor
import functions as F


class MSEloss:
    """
    Criterion that measures the mean squared error
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self, preds, target):
        return F.MSEloss(preds, target)


class CrossEntropyLoss:
    """
    Cross entropy loss
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self, gts, target):
        raise NotImplementedError
