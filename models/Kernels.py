from abc import ABC, abstractmethod
import numpy as np
class Kernel(ABC):

    @abstractmethod
    def __call__(self,X,y):
        pass

class RBFKernel(Kernel):

    def __init__(self,gamma_rbf):
        self.gamma_rbf = gamma_rbf
    def __call__(self,X,y):
        dists=np.sum((X-y)**2,axis=1)
        return np.exp(-self.gamma_rbf * dists)

class LinearKernel(Kernel):
    def __init__(self):
        pass
    def __call__(self,X,y):
        return np.dot(X,y)