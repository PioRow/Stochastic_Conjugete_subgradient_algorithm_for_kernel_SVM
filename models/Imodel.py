from abc import ABC, abstractmethod
import numpy as np
from .Kernels import *

class Imodel(ABC):
    def __init__(self, kernel=None):
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = RBFKernel(0.1)
            
    @abstractmethod
    def predict(self, X):
        pass

    @staticmethod
    def eval_f(alpha, Q, W):
        return 0.5 * alpha @ Q @ alpha + np.mean(np.maximum(0, 1 - W * (Q @ alpha)))

    @staticmethod
    def eval_f_scores(alpha, g, W):
        return 0.5 * np.dot(alpha, g) + np.mean(np.maximum(0, 1 - W * g))
    
    def _compute_kernel_row(self, X, y):
        return self.kernel(X, y)