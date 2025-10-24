import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#DMD as used by rolling forecast approach
def dmd_decomposition_rolling(X1, X2):
    U, Sigma, Vt   = np.linalg.svd(X1, full_matrices=False)
    Sigma_inv      = np.diag(1.0 / Sigma)
    A_tilde        = U.T @ X2 @ Vt.T @ Sigma_inv
    Lambda, W      = np.linalg.eig(A_tilde)

    # Regularize near-zero eigenvalues
    epsilon = 1e-8
    Lambda  = np.where(np.abs(Lambda) < epsilon, epsilon, Lambda)
    Phi     = U @ W

    # b_last from the last snapshot in X1
    x_last = X1[:, -1]  # shape (n,)
    b_last = np.linalg.pinv(Phi) @ x_last
    return Phi, Lambda, b_last


#DMD for single step usage (probably totally obsolete)
def dmd_decomposition_single_step(X1, X2):
    U, Sigma, Vt = np.linalg.svd(X1, full_matrices=False)
    Sigma_inv    = np.diag(1.0 / Sigma)
    A_tilde      = U.T @ X2 @ Vt.T @ Sigma_inv
    Lambda, W    = np.linalg.eig(A_tilde)

    # Regularize near-zero eigenvalues
    epsilon = 1e-8
    Lambda  = np.where(np.abs(Lambda) < epsilon, epsilon, Lambda)
    Phi     = U @ W

    # Continuous-time eigenvalues
    omega = np.log(Lambda)

    # Mode amplitudes from the *first* snapshot
    x0 = X1[:, 0]
    b  = np.linalg.pinv(Phi) @ x0
    return Phi, Lambda, omega, b