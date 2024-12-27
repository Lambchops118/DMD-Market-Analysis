#functions that deal with rolling forecast stuff
import numpy as np
import matplotlib.pyplot as plt

import data_preparation
import dmd_functions

def dmd_predict(Phi, Lambda, b_last):
    return Phi @ (np.diag(Lambda) @ b_last)


#Compare signs of actual vs predicted next value. Returns fraction of correct sign predictions (between 0 and 1)
def evaluate_directional_success(actual, predicted):

    # print("Actual:")
    # print(actual)
    # print("Predicted:")
    # print(predicted.real)
    #
    # print("np mean np sign actual")
    # print(np.mean(np.sign(actual)))
    # print("np sign predicted")
    # print(np.sign(predicted))

    #input()
    #editing this for only the real parts, as imaginary parts may be screwing up comparison
    return np.mean(np.sign(actual) == np.sign(predicted.real))

    #return np.mean(np.sign(actual) == np.sign(predicted))


#rolling walk forecast across entire dataset for multiple m, ell combos
def evaluate_hot_spots_rolling(file_paths, m_values, ell_values, threshold=0.5):
    data_matrix = data_preparation.load_log_return_matrix(file_paths)  # shape (n, T)
    n, T = data_matrix.shape
    results = {}

    for m in m_values:
        for ell in ell_values:
            successes = 0
            total_checks = 0

            # t: start index of the window
            # last day in the window: t+m-1
            # forecast day: (t+m-1) + ell
            # => we must have t+m-1+ell <= T-1 => t <= T - m - ell
            for t in range(0, T - m - ell + 1):
                X1 = data_matrix[:, t : t + m - 1]  # shape (n, m-1)
                X2 = data_matrix[:, t + 1 : t + m]  # shape (n, m-1)
                if X1.shape[1] < 1:
                    continue

                try:
                    Phi, Lambda, b_last = dmd_functions.dmd_decomposition_rolling(X1, X2)
                    # Repeated application of diag(Lambda) for 'ell' steps
                    c0 = b_last
                    Lambda_ell = Lambda ** ell
                    c_ell = Lambda_ell * c0
                    predicted_log_return = Phi @ c_ell  # shape (n,)

                    #debug for transient eigenvalues
                    predicted_log_return = predicted_log_return.real

                    actual_day = t + m - 1 + ell
                    if actual_day < T:
                        actual_log_return = data_matrix[:, actual_day]  # shape (n,)
                        sr = evaluate_directional_success(actual_log_return,
                                                           predicted_log_return)
                        successes += sr * n  # sr is fraction among n cryptos
                        total_checks += n

                        #print("predicted log return in value")
                        #print(predicted_log_return)

                        #print("actual log return in value ")
                        #print(actual_log_return)
                        #input()

                except np.linalg.LinAlgError:
                    # Skip if rank-deficient
                    pass

            if total_checks == 0:
                # No valid windows
                continue

            overall_success_rate = successes / total_checks
            results[(m, ell)] = overall_success_rate

    # hot_spots
    hot_spots = {k: v for k, v in results.items() if v >= threshold}
    return results, hot_spots

def plot_hot_spots(results):
    if not results:
        print("No results to plot.")
        return

    m_values = sorted(set(k[0] for k in results.keys()))
    ell_values = sorted(set(k[1] for k in results.keys()))

    # Build heatmap array
    heatmap = np.zeros((len(m_values), len(ell_values)))
    for (m, ell), rate in results.items():
        i = m_values.index(m)
        j = ell_values.index(ell)
        heatmap[i, j] = rate

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar(label="Success Rate")
    plt.xticks(range(len(ell_values)), ell_values, rotation=45)
    plt.yticks(range(len(m_values)), m_values)
    plt.xlabel("Prediction Horizon (ell)")
    plt.ylabel("Window Size (m)")
    plt.title("Rolling Forecast: Success Rate Heatmap")
    plt.tight_layout()
    plt.show()