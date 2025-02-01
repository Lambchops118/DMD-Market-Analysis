import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#predict next LOG return after last columns of X1
def predict_one_step_beyond_last_snapshot(X1, Phi, Lambda):
    x_last = X1[:, -1]  # shape (n,)
    c_last = np.linalg.pinv(Phi) @ x_last
    x_next = Phi @ (np.diag(Lambda) @ c_last)
    return x_next

def predict_ell_steps_beyond_last_snapshot(X1, Phi, Lambda, ell):
    """
    Predict 'ell' steps beyond the last column of X1 using DMD.

    X1 : array of shape (n, T)
        The dataset (e.g., log returns) up to time T.
    Phi : array of shape (n, r)
        The DMD modes.
    Lambda : array of shape (r,)
        The DMD eigenvalues (one per mode).
    ell : int
        How many steps forward to predict.
    """
    # x_last is the final snapshot (state) in X1
    x_last = X1[:, -1]  # shape (n,)

    # Project x_last onto the DMD modes
    c_last = np.linalg.pinv(Phi) @ x_last  # shape (r,)

    # Raise each eigenvalue to the power ell (Lambda^ell)
    Lambda_ell = np.diag(Lambda ** ell)    # shape (r, r)

    # Reconstruct the predicted state
    x_ell = Phi @ (Lambda_ell @ c_last)    # shape (n,)

    return x_ell