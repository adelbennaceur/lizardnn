from tensor import Tensor


class MSEloss():
    """
    Criterion that measures the mean squared error
    """ 
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self ,preds, target):
        return ((preds-target)*(preds-target).sum(0)) 
    

class CrossEntropyLoss():
    """
    Criterion that measures the mean squared error
    """ 
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self , gts, target):
        raise NotImplementedError    
        

