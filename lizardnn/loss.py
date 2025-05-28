from lizardnn.tensor import Tensor


class Loss:
    """Base class for all loss functions."""

    def __init__(self, reduction: str = "mean") -> None:
        """Initialize loss function.

        Args:
            reduction: Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'
        """
        assert reduction in ["none", "mean", "sum"]
        self.reduction = reduction

    def __call__(self, *args, **kwargs) -> Tensor:
        """Forward pass of the loss function."""
        return self.forward(*args, **kwargs)


class MSELoss(Loss):
    """Mean Squared Error (L2) loss.

    This creates a criterion that measures the mean squared error between
    each element in the input x and target y.
    """

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """Compute MSE loss.

        Args:
            preds: Predictions from model
            target: Ground truth values

        Returns:
            Loss value
        """

        diff = preds - target
        mse = (diff * diff).mean()
        if self.reduction == "sum":
            return mse * diff.data.size
        elif self.reduction == "none":
            return diff * diff
        return mse


class L1Loss(Loss):
    """Mean Absolute Error (L1) loss.

    Creates a criterion that measures the mean absolute error between
    each element in the input x and target y.
    """

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """Compute L1 loss.

        Args:
            preds: Predictions from model
            target: Ground truth values

        Returns:
            Loss value
        """
        diff = preds - target
        loss = abs(diff)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CrossEntropyLoss(Loss):
    """Cross entropy loss with LogSoftmax.

    This criterion combines LogSoftmax and NLLLoss in a single class.
    Useful for training a classification problem with C classes.
    """

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """Compute cross entropy loss.

        Args:
            preds: Raw predictions from model (logits)
            target: Ground truth class indices or class probabilities

        Returns:
            Loss value
        """
        return preds.cross_entropy(target)


class BCELoss(Loss):
    """Binary Cross Entropy loss.

    Creates a criterion that measures the Binary Cross Entropy
    between the target and the input probabilities.
    """

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """Compute binary cross entropy loss.

        Args:
            preds: Predictions from model (after sigmoid)
            target: Ground truth values (0 or 1)

        Returns:
            Loss value
        """

        eps = 1e-7
        one = Tensor(1.0, requires_grad=False)

        loss = -(
            target * (preds + eps).log() + (one - target) * (one - preds + eps).log()
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
