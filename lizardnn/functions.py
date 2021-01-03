from tensor import Tensor


def cross_entropy(preds, target):

    return None


def MSEloss(preds, target):
    return (preds - target) * (preds - target).sum(0)
