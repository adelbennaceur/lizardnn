
import numpy as np 



class Tensor(object):

    def __init__(self , data , ):
        self.data = np.array(data)
    
    def __repr__(self):
        return str(self.data.__repr__())    

    def __add__(self, other):
        return Tensor(self.data + other.data)
    
    def __mul__(self, other):
        return Tensor(self.data*other.data)
    
    def __sub__(self,other):
        return Tensor(self.data-other.data)




if __name__ == "__main__":
    x = Tensor([1,2,3,4,5])
    y = Tensor([0,5,4,8])

    print(y*y)

