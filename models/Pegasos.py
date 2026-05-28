from .Kernels import *

from .Imodel import Imodel

class PegasosBaseline(Imodel):
    def __init__(self, kernel=None, gamma_rbf=0.1, verbose=False):
        super().__init__(kernel)
        if not kernel:
            self.kernel = RBFKernel(gamma_rbf)
        else:
            self.kernel = kernel
        self.gamma_rbf = gamma_rbf
        self.verbose = verbose
        # hardcoded lambda_param for Pegasos
        self.lambda_param = 1.0
        self.history = []
        self.alpha = None
        self.S_k = None
        self.W_k = None

    def fit(self, X_train, y_train, max_iter=100, precompute_kernel=False):
        m_samples, n_features = X_train.shape
        self.S_k = X_train
        self.W_k = y_train

        # Only precompute the full kernel matrix if explicitly requested, avoiding problems with large datasets
        if precompute_kernel:
            Q = np.zeros((m_samples, m_samples))
            for i in range(m_samples):
                Q[i] = self._compute_kernel_row(X_train, X_train[i])
        else:
            Q = None

        alpha_count = np.zeros(m_samples)

        for t in range(1, max_iter + 1):
            eta_t = 1.0 / (self.lambda_param * t)

            i = np.random.randint(0, m_samples)

            # from precomputed Q or compute on the fly
            Q_i = Q[i] if precompute_kernel else self._compute_kernel_row(X_train, X_train[i])

            inner = eta_t * np.sum(alpha_count * self.W_k * Q_i)
            margin = self.W_k[i] * inner

            if margin < 1:
                alpha_count[i] += 1
            if precompute_kernel:
                tmp_alpha = (1.0 / (self.lambda_param * t)) * alpha_count * self.W_k
                f_val = self.eval_f(tmp_alpha, Q, self.W_k)
                self.history.append(f_val)
            # evaluate the objective function if the full kernel matrix is available
            if self.verbose and precompute_kernel and t % max(1, max_iter // 10) == 0:
                tmp_alpha = (1.0 / (self.lambda_param * t)) * alpha_count * self.W_k
                f_val = self.eval_f(tmp_alpha, Q, self.W_k)
                print(f"[Pegasos] iter {t}/{max_iter}  f={f_val:.4f}  "
                      f"nnz={int(np.sum(alpha_count > 0))}")

        self.alpha = (1.0 / (self.lambda_param * max_iter)) * alpha_count * self.W_k

    def predict(self, X_new):
        if self.alpha is None or self.S_k is None:
            raise ValueError("The model must be trained via fit() before making predictions.")

        predictions = []
        for x_test in X_new:
            kernel_row = self._compute_kernel_row(self.S_k, x_test)
            score = np.dot(self.alpha, kernel_row)
            predictions.append(1 if score >= 0 else -1)
        return np.array(predictions)

    @staticmethod
    def eval_f(alpha, Q, W):
        return 0.5 * alpha @ Q @ alpha + np.mean(np.maximum(0, 1 - W * (Q @ alpha)))

    def _compute_kernel_row(self, X, y):
        return self.kernel(X, y)