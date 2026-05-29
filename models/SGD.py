from .Kernels import *

from .Imodel import Imodel

class SGDBaseline(Imodel):
    def __init__(self, kernel=None, gamma_rbf=0.1, eta=0.01, verbose=False):
        super().__init__(kernel)
        if not kernel:
            self.kernel = RBFKernel(gamma_rbf)
        else:
            self.kernel = kernel
        self.gamma_rbf = gamma_rbf
        self.eta = eta
        self.verbose = verbose
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

        alpha = np.zeros(m_samples)

        for t in range(1, max_iter + 1):
            i = np.random.randint(0, m_samples)

            # Retrieve the precomputed row or compute it on the fly
            Q_i = Q[i] if precompute_kernel else self._compute_kernel_row(X_train, X_train[i])

            inner = np.dot(alpha, Q_i)
            margin = self.W_k[i] * inner

            # Calculate the exact gradient of the quadratic term
            # Warning computing this on the fly without precomputed Q is very slow
            if precompute_kernel:
                grad_quad = Q @ alpha
            else:
                grad_quad = np.array([np.dot(alpha, self._compute_kernel_row(X_train, x)) for x in X_train])

            if margin < 1:
                # Add hinge loss subgradient for active margin
                gradient = grad_quad - self.W_k[i] * Q_i
            else:
                gradient = grad_quad

            alpha = alpha - self.eta * gradient
            if precompute_kernel:
                f_val = self.eval_f(alpha, Q, self.W_k)
                self.history.append(f_val)
            # Only evaluate the objective function if the full kernel matrix is available
            if self.verbose and precompute_kernel and t % max(1, max_iter // 10) == 0:
                f_val = self.eval_f(alpha, Q, self.W_k)
                print(f"[SGD] iter {t}/{max_iter}  f={f_val:.4f}")

        self.alpha = alpha

    def predict(self, X_new):
        if self.alpha is None or self.S_k is None:
            raise ValueError("The model must be trained via fit() before making predictions.")

        predictions = []
        for i, x_test in enumerate(X_new):
            
            kernel_row = self._compute_kernel_row(self.S_k, x_test)
            score = np.dot(self.alpha, kernel_row)
            predictions.append(1 if score >= 0 else -1)
        return np.array(predictions)

