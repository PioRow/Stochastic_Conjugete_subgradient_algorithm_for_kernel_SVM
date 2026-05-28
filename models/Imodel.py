from abc import ABC, abstractmethod
import numpy as np
from .Kernels import *

class Imodel:

    def __init__(self, kernel:Kernel=None):
        if kernel is None:
            self.kernel = RBFKernel(0.1)
        self.kernel = kernel
    def predict(self,X):
        pass
    @staticmethod
    def eval_f(alpha, Q, W):
        return 0.5 * alpha @ Q @ alpha + np.mean(np.maximum(0, 1 - W * (Q @ alpha)))

    def _compute_kernel_row(self, X, y):
        return self.kernel(X, y)