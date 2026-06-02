import numpy as np
from .Kernels import *

from .Imodel import Imodel
class StochasticConjugateSubgradientAlgorithm(Imodel):
    def __init__(self, kernel=None, epsilon=1e-3, delta_min=1e-4, delta_max=5.0,
                 m1=0.3, m2=0.25, ls_n=5, eta_1=0.3, eta_2=0.1, gamma=1.2, 
                 verbose=False):
        super().__init__(kernel=kernel)
        self.epsilon = epsilon
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.alpha = None
        self.S_k = None
        self.W_k = None
        self.S_left_idx=None
        self.m1=m1
        self.m2=m2
        self.ls_n=ls_n
        self.verbose=verbose
        self.eta_1 = eta_1
        self.eta_2 = eta_2
        self.gamma=gamma
        self.history = []

    def fit(self,X_train,y_train,max_iter=100,batch_size=10, record_history=False):
        m_samples,n_features = X_train.shape
        self.S_left_idx=np.arange(m_samples)
        x_sample,y_sample=self._sample_ds(X_train,y_train,batch_size)
        self.S_k=x_sample
        self.W_k=y_sample
        Q_k=np.zeros((batch_size,batch_size))

        for i in range(batch_size):
            Q_k[i]=self._compute_kernel_row(x_sample,x_sample[i])
        alpha_hat=np.zeros(batch_size)
        d_k=self._compute_subgradient(alpha_hat,Q_k,self.W_k)
        delta_k=1.0
        for k in range(1,max_iter):
            norm_d=np.linalg.norm(d_k)

            if norm_d <self.epsilon and delta_k<self.delta_min:
                break

            g_k=self._compute_subgradient(alpha_hat,Q_k,self.W_k)
            d_vec=-self._nr_operation(d_k,g_k)

            f_curr=self.eval_f(alpha_hat,Q_k,self.W_k)

            t_k=self.line_search(f_curr,alpha_hat,d_vec,Q_k,self.W_k,delta_k)
            alpha_hat_new=alpha_hat+t_k*d_vec

            x_new,y_new=self._sample_ds(X_train,y_train,batch_size)
            if x_new is None:
                print("Dataset fully consumed. Ending training loop.")
                break
            curr_m = len(self.S_k)+batch_size
            T_x, T_w = self._sample_ds(X_train, y_train, curr_m)
            if T_x is None:
                print("Insufficient samples remaining to construct validation batch T_k.")
                break
            self.S_k = np.vstack([self.S_k, x_new])
            self.W_k = np.concatenate([self.W_k, y_new])



            new_Q = np.zeros((curr_m, curr_m))
            new_Q[:curr_m - batch_size, :curr_m - batch_size] = Q_k
            for i in range(curr_m - batch_size, curr_m):
                row = self._compute_kernel_row(self.S_k, self.S_k[i])
                new_Q[i, :] = row
                new_Q[:, i] = row
            Q_k = new_Q



            Q_hat_k = np.zeros((curr_m, curr_m))
            for i in range(curr_m):
                Q_hat_k[i] = self._compute_kernel_row(T_x, T_x[i])
            beta_k = np.concatenate([alpha_hat_new, np.zeros(batch_size)])
            beta_hat_prev = np.concatenate([alpha_hat, np.zeros(batch_size)])
            d_vec_new = np.concatenate([d_vec, np.zeros(batch_size)])
            f_k_beta = self.eval_f(beta_k, Q_k, self.W_k)
            f_k_beta_hat = self.eval_f(beta_hat_prev, Q_k, self.W_k)

            f_val_beta = self.eval_f(beta_k, Q_hat_k, T_w)
            f_val_beta_hat = self.eval_f(beta_hat_prev, Q_hat_k, T_w)
            lhs_cond = (f_k_beta - f_k_beta_hat) <= self.eta_1 * (f_val_beta - f_val_beta_hat)
            rhs_cond = np.linalg.norm(d_vec_new) > self.eta_2 * delta_k
            if lhs_cond and rhs_cond:
                alpha_hat = beta_k
                delta_k = min(self.gamma * delta_k, self.delta_max)
            else:
                alpha_hat = beta_hat_prev
                delta_k = max(delta_k / self.gamma, self.delta_min)
            d_k = d_vec_new
            self.history.append(self.eval_f(alpha_hat, Q_k, self.W_k))
        self.alpha = alpha_hat

    def predict(self, X_new):
        if self.alpha is None or self.S_k is None:
            raise ValueError("The model must be trained via fit() before making predictions.")

        predictions = []
        for x_test in X_new:
            kernel_row = self._compute_kernel_row(self.S_k, x_test)
            score = np.dot(self.alpha, kernel_row)
            predictions.append(1 if score >= 0 else -1)
        return np.array(predictions)

    def line_search(self,f_val, alpha, d, Q, W, delta_k):
        def check_intersect(step_t):
            alpha_new=alpha+step_t*d
            f_new=self.eval_f(alpha_new,Q,W)
            g_new=self._compute_subgradient(alpha_new,Q,W)
            int_L=(f_new-f_val)<=-self.m1*np.dot(d,d)*step_t
            dir_der=np.dot(g_new,d)
            int_R=(0<dir_der) and (dir_der>=self.m2*np.dot(d,d))
            return int_L,int_R
        b = delta_k / self.ls_n
        norm_d = np.linalg.norm(d)
        t = delta_k / norm_d if norm_d > 1e-9 else delta_k

        int_L,int_R = check_intersect(t)
        if int_L and int_R:
            return t
        elif int_L and not int_R:
            while int_L and not int_R and (t * norm_d <= delta_k):
                t=2*t
                int_L,int_R = check_intersect(t)
            if t*norm_d >delta_k:
                return t/2.
            low,high=t/2.0,t
        else: # not int_L and int_R
            while ( not int_L) and (t*norm_d >= b):
                t=t/2.0
                int_L,int_R = check_intersect(t)
            if int_L and int_R:
                return t
            if t*norm_d<b:
                return 0.0
            low,high=t,2*t

        int_L,int_R = check_intersect(t)
        if int_L and int_R:
            return t
        for _ in range(20):
            t=(low+high)/2.0
            int_L,int_R = check_intersect(t)
            if int_L and int_R:
                return t
            if int_R and not int_L:
                high=t
            else:
                low=t
        return t


    def _nr_operation(self,d_prev,g_curr):
        v=g_curr+d_prev
        norm_v_sq=np.dot(v,v)
        if norm_v_sq<1e-9:
            return g_curr
        lam=-np.dot(-d_prev,v)/norm_v_sq
        lam_str=np.clip(lam,0,1)
        return lam_str* (-d_prev)+(1-lam_str)*g_curr

    def _compute_kernel_row(self,X,y):
        return self.kernel(X,y)

    #TODO
    # -Grad(f(x_k)):=D_K ,d_{K-1}-> B_k d_{k-1}
    # B_k=norm(D_k)**2/(D_K'(D_{k}-D_{k-1})
    #
    def _compute_subgradient(self,alpha,Q,W):
        m=len(W)
        quad_g=Q @ alpha
        margins=1-W*(quad_g)
        acitve_check= margins>0
        hinge_g=np.zeros_like(alpha)
        if np.any(acitve_check):
            hinge_g=-np.sum(W[acitve_check][:,None]*Q[acitve_check],axis=0)/m
        return hinge_g+quad_g

    def _sample_ds(self,X,y,batch_size):

        if len(self.S_left_idx)<batch_size:
            return None,None
        choice_idx = np.random.choice(
            len(self.S_left_idx),
            batch_size,
            replace=False
        )
        ds_idx=self.S_left_idx[choice_idx]
        self.S_left_idx=np.delete(self.S_left_idx,choice_idx)
        return X[ds_idx],y[ds_idx]



 

