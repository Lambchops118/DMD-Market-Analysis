import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#predict next LOG return after last columns of X1
def predict_one_step_beyond_last_snapshot(X1, Phi, Lambda):
    x_last = X1[:, -1]  # shape (n,)
    c_last = np.linalg.pinv(Phi) @ x_last
    x_next = Phi @ (np.diag(Lambda) @ c_last)
    return x_next