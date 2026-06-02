from .Kernels import *
from .Imodel import Imodel
import numpy as np

class PegasosBaseline(Imodel):
    def __init__(self, kernel=None, verbose=False):
        super().__init__(kernel=kernel)
        self.verbose = verbose
        # hardcoded lambda_param for Pegasos
        self.lambda_param = 1.0
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

        alpha_count = np.zeros(m_samples)
        # u tracks Q @ c, with c = alpha_count * W (unscaled state)
        u = np.zeros(m_samples)
        self.history = []

        for t in range(1, max_iter + 1):
            eta_t = 1.0 / (self.lambda_param * t)
            i = np.random.randint(0, m_samples)
            Q_i = Q[i] if precompute_kernel else self._compute_kernel_row(X_train, X_train[i])
            # margin = eta_t * (Q c)_i  == score g_i; c = alpha_count * W
            inner = eta_t * u[i]
            
            if W[i] * inner < 1:
                alpha_count[i] += 1
                # incremental update, as a row is already computed
                u += W[i] * Q_i

            if record_history or (self.verbose and t % max(1, max_iter // 10) == 0):
                c = alpha_count * W
                f_val = self.eval_f_scores(eta_t * c, eta_t * u, W)
                
                if record_history:
                    self.history.append(f_val)
                    
                if self.verbose and t % max(1, max_iter // 10) == 0:
                    print(f"[Pegasos] iter {t}/{max_iter}  f={f_val:.4f}  "
                          f"nnz={int(np.sum(alpha_count > 0))}")

        # scale the final vector once at the very end
        self.alpha = (1.0 / (self.lambda_param * max_iter)) * alpha_count * W

    def predict(self, X_new):
        if self.alpha is None or self.S_k is None:
            raise ValueError("The model must be trained via fit() before making predictions.")

        predictions = []
        for x_test in X_new:
            kernel_row = self._compute_kernel_row(self.S_k, x_test)
            score = np.dot(self.alpha, kernel_row)
            predictions.append(1 if score >= 0 else -1)
        return np.array(predictions)