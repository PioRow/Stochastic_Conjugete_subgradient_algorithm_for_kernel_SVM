import numpy as np
from .Kernels import *
from .Imodel import Imodel

class SGDBaseline(Imodel):
    def __init__(self, kernel=None, eta=0.01, verbose=False):
        super().__init__(kernel=kernel)
        self.eta = eta
        self.verbose = verbose
        self.history = []
        self.alpha = None
        self.S_k = None
        self.W_k = None

    def fit(self, X_train, y_train, max_iter=100, precompute_kernel=False, record_history=False):
        m_samples, n_features = X_train.shape
        self.S_k, self.W_k = X_train, y_train
        W = self.W_k

        Q = None
        if precompute_kernel:
            Q = np.zeros((m_samples, m_samples))
            for i in range(m_samples):
                Q[i] = self._compute_kernel_row(X_train, X_train[i])

        self.alpha = np.zeros(m_samples)
        
        u = np.zeros(m_samples) 
        self.history = []

        for t in range(1, max_iter + 1):
            i = np.random.randint(0, m_samples)

            # fetch or compute the i-th row of the kernel matrix
            Q_i = Q[i] if precompute_kernel else self._compute_kernel_row(X_train, X_train[i])
            margin = W[i] * u[i]
            self.alpha = (1 - self.eta) * self.alpha
            u = (1 - self.eta) * u
            if margin < 1:
                self.alpha[i] += self.eta * W[i]
                # keep u = Q @ alpha
                u += self.eta * W[i] * Q_i

            if record_history or (self.verbose and t % max(1, max_iter // 10) == 0):
                f_val = self.eval_f_scores(self.alpha, u, W)
                if record_history:
                    self.history.append(f_val)
                    
                if self.verbose and t % max(1, max_iter // 10) == 0:
                    print(f"[SGD] iter {t}/{max_iter}  f={f_val:.4f}")

    def predict(self, X_new):
        if self.alpha is None or self.S_k is None:
            raise ValueError("The model must be trained via fit() before making predictions.")
        predictions = []
        for x_test in X_new:
            kernel_row = self._compute_kernel_row(self.S_k, x_test)
            score = np.dot(self.alpha, kernel_row)
            predictions.append(1 if score >= 0 else -1)
        return np.array(predictions)