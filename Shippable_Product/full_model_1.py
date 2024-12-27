import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##############################################################################
#                            DATA LOADING FUNCTIONS                          #
##############################################################################
def load_log_return_matrix(file_paths):
    """
    Load multiple cryptocurrency CSVs, align them by date, compute daily log returns,
    and produce a single matrix of shape (n, T).
    - Each row i is the log-return series of crypto i (n cryptos),
      truncated so they all share the same length T.
    - We align by the LATEST common start date among all cryptos.

    Parameters
    ----------
    file_paths : list of str
        Paths to CSV files, each containing 'snapped_at' and 'price' columns.

    Returns
    -------
    data_matrix : np.ndarray, shape (n, T)
        Log-return matrix, where row i corresponds to one crypto.
    """
    all_data = []
    start_dates = []
    original_prices = []

    # 1) Read each file, record earliest data, compute daily means, store raw prices
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        if 'snapped_at' not in df.columns or 'price' not in df.columns:
            raise ValueError(f"File {file_path} missing 'snapped_at' or 'price'.")

        df['snapped_at'] = pd.to_datetime(df['snapped_at'])
        df = df.sort_values(by='snapped_at')
        df_daily = df.set_index('snapped_at').resample('D').mean()

        start_dates.append(df_daily.index.min())

        prices = df_daily['price'].dropna().values
        original_prices.append(prices)

    # 2) Align from the latest common start date
    common_start_date = max(start_dates)

    aligned_data = []
    for file_path, orig_price in zip(file_paths, original_prices):
        df = pd.read_csv(file_path)
        df['snapped_at'] = pd.to_datetime(df['snapped_at'])
        df = df.sort_values(by='snapped_at')
        df_daily = df.set_index('snapped_at').resample('D').mean()
        df_daily = df_daily[df_daily.index >= common_start_date]
        prices = df_daily['price'].dropna().values

        if len(prices) > 1:
            log_returns = np.log(prices[1:] / prices[:-1])
        else:
            log_returns = np.array([])

        aligned_data.append(log_returns)

    # 3) Truncate all to the same length
    min_length = min(len(lr) for lr in aligned_data if len(lr) > 0)
    truncated_data = [lr[:min_length] for lr in aligned_data]

    # 4) Stack into (n, T)
    data_matrix = np.array(truncated_data)
    return data_matrix  # shape (n, T)


def load_and_prepare_data_single_step(file_paths):
    """
    Load cryptocurrency data from CSV files, align them by date, compute daily log returns,
    and form (X1, X2) matrices for single-step DMD. This is a more "direct" single-step approach.

    Parameters
    ----------
    file_paths : list of str
        CSV paths with 'snapped_at' (date) and 'price'.

    Returns
    -------
    X1 : np.ndarray, shape (n, T-1)
        Each row: time series of log returns for one crypto, excluding the last time step.
    X2 : np.ndarray, shape (n, T-1)
        Same as X1, but shifted one step forward in time.
    truncated_prices : list of np.ndarray
        The aligned daily price series for each crypto, each truncated to the same length.
        truncated_prices[i] has length T (one more than the log returns).
    """
    all_data = []
    original_prices = []
    start_dates = []

    # 1) First pass: read each file, store earliest date among all cryptos
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        if 'snapped_at' not in df.columns or 'price' not in df.columns:
            raise ValueError(f"File {file_path} missing 'snapped_at' or 'price' columns.")

        df['snapped_at'] = pd.to_datetime(df['snapped_at'])
        df = df.sort_values(by='snapped_at')
        df_daily = df.set_index('snapped_at').resample('D').mean()

        start_dates.append(df_daily.index.min())

        prices = df_daily['price'].dropna().values
        original_prices.append(prices)

        if len(prices) > 1:
            log_returns = np.log(prices[1:] / prices[:-1])
        else:
            log_returns = np.array([])
        all_data.append(log_returns)

    # 2) Determine the latest common start date
    common_start_date = max(start_dates)

    # 3) Align from that common start date forward
    aligned_data = []
    aligned_prices = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df['snapped_at'] = pd.to_datetime(df['snapped_at'])
        df = df.sort_values(by='snapped_at')
        df_daily = df.set_index('snapped_at').resample('D').mean()
        df_daily = df_daily[df_daily.index >= common_start_date]
        prices = df_daily['price'].dropna().values

        if len(prices) > 1:
            log_returns = np.log(prices[1:] / prices[:-1])
        else:
            log_returns = np.array([])

        aligned_data.append(log_returns)
        aligned_prices.append(prices)

    # 4) Truncate all log-return series to the same minimum length
    min_length = min(len(lr) for lr in aligned_data if len(lr) > 0)
    truncated_data = [lr[:min_length] for lr in aligned_data]
    truncated_prices = [p[:min_length + 1] for p in aligned_prices]

    # 5) Stack into data_matrix
    data_matrix = np.array(truncated_data)  # shape (n, T)

    # 6) Build X1, X2 for single-step shift
    X1 = data_matrix[:, :-1]  # shape (n, T-1)
    X2 = data_matrix[:, 1:]   # shape (n, T-1)

    return X1, X2, truncated_prices


##############################################################################
#                            DMD FUNCTIONS                                   #
##############################################################################
def dmd_decomposition_rolling(X1, X2):
    """
    DMD decomposition used by the rolling forecast approach.
    Returns (Phi, Lambda, b_last), where b_last is derived from the *last*
    snapshot of X1.

    Parameters
    ----------
    X1, X2 : np.ndarray
        Each shape (n, k), representing consecutive snapshots in time.

    Returns
    -------
    Phi : np.ndarray, shape (n, r)
        DMD modes.
    Lambda : np.ndarray, shape (r,)
        Eigenvalues.
    b_last : np.ndarray, shape (r,)
        Mode amplitudes for the LAST snapshot in X1.
    """
    U, Sigma, Vt = np.linalg.svd(X1, full_matrices=False)
    Sigma_inv = np.diag(1.0 / Sigma)
    A_tilde = U.T @ X2 @ Vt.T @ Sigma_inv

    Lambda, W = np.linalg.eig(A_tilde)

    # Regularize near-zero eigenvalues
    epsilon = 1e-8
    Lambda = np.where(np.abs(Lambda) < epsilon, epsilon, Lambda)

    Phi = U @ W

    # b_last from the last snapshot in X1
    x_last = X1[:, -1]  # shape (n,)
    b_last = np.linalg.pinv(Phi) @ x_last
    return Phi, Lambda, b_last


def dmd_decomposition_single_step(X1, X2):
    """
    Standard DMD decomposition for single-step usage example.
    Returns (Phi, Lambda, omega, b), where b is from the *first* snapshot of X1
    and omega = log(Lambda) for continuous-time eigenvalues.

    Parameters
    ----------
    X1, X2 : np.ndarray, shape (n, T-1)

    Returns
    -------
    Phi : np.ndarray, shape (n, r)
    Lambda : np.ndarray, shape (r,)
    omega : np.ndarray, shape (r,)
    b : np.ndarray, shape (r,)
    """
    U, Sigma, Vt = np.linalg.svd(X1, full_matrices=False)
    Sigma_inv = np.diag(1.0 / Sigma)
    A_tilde = U.T @ X2 @ Vt.T @ Sigma_inv

    Lambda, W = np.linalg.eig(A_tilde)

    # Regularize near-zero eigenvalues
    epsilon = 1e-8
    Lambda = np.where(np.abs(Lambda) < epsilon, epsilon, Lambda)

    Phi = U @ W

    # Continuous-time eigenvalues
    omega = np.log(Lambda)

    # Mode amplitudes from the *first* snapshot
    x0 = X1[:, 0]
    b = np.linalg.pinv(Phi) @ x0
    return Phi, Lambda, omega, b


##############################################################################
#                        ROLLING FORECAST-RELATED                            #
##############################################################################
def dmd_predict(Phi, Lambda, b_last):
    """
    One-step DMD forecast from the last snapshot in X1
    (used in the rolling forecast procedure).

    Parameters
    ----------
    Phi : np.ndarray (n x r)
    Lambda : np.ndarray (r,)
    b_last : np.ndarray (r,)

    Returns
    -------
    x_next : np.ndarray (n,)
        One-step ahead forecast in original variable space.
    """
    return Phi @ (np.diag(Lambda) @ b_last)


def evaluate_directional_success(actual, predicted):
    """
    Compare signs of 'actual' vs. 'predicted' (both shape (n,)).
    Return fraction of matching signs.

    Parameters
    ----------
    actual : np.ndarray
    predicted : np.ndarray

    Returns
    -------
    float
        Fraction of correct sign predictions (between 0 and 1).
    """
    return np.mean(np.sign(actual) == np.sign(predicted))


def evaluate_hot_spots_rolling(file_paths, m_values, ell_values, threshold=0.5):
    """
    Rolling (walk-forward) forecast across the entire dataset for multiple
    (m, ell) combinations. We store success rates in a results dictionary.

    The data is loaded as a log-return matrix of shape (n, T).
    A 'window' of length m means we take X1 = data[:, t : t+m-1] and
    X2 = data[:, t+1 : t+m], each with shape (n, m-1).

    Then we forecast ell steps ahead by repeatedly applying diag(Lambda).

    Parameters
    ----------
    file_paths : list of str
    m_values : iterable of int
        Window sizes
    ell_values : iterable of int
        Forecast horizons (how many steps ahead)
    threshold : float
        Minimum success rate for calling a (m,ell) a 'hot spot'

    Returns
    -------
    results : dict
        {(m, ell) : success_rate}
    hot_spots : dict
        Subset of results where success_rate >= threshold
    """
    data_matrix = load_log_return_matrix(file_paths)  # shape (n, T)
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
                    Phi, Lambda, b_last = dmd_decomposition_rolling(X1, X2)
                    # Repeated application of diag(Lambda) for 'ell' steps
                    c0 = b_last
                    Lambda_ell = Lambda ** ell
                    c_ell = Lambda_ell * c0
                    predicted_log_return = Phi @ c_ell  # shape (n,)

                    actual_day = t + m - 1 + ell
                    if actual_day < T:
                        actual_log_return = data_matrix[:, actual_day]  # shape (n,)
                        sr = evaluate_directional_success(actual_log_return,
                                                           predicted_log_return)
                        successes += sr * n  # sr is fraction among n cryptos
                        total_checks += n
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
    """
    Plot a heatmap of success rates over (m, ell).

    Parameters
    ----------
    results : dict
        {(m, ell) : success_rate}
    """
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


##############################################################################
#                       SINGLE-STEP FORECAST-RELATED                         #
##############################################################################
def predict_one_step_beyond_last_snapshot(X1, Phi, Lambda):
    """
    Predict the next log-return one step AFTER the last column of X1.

    Parameters
    ----------
    X1 : np.ndarray, shape (n, T-1)
    Phi : np.ndarray, shape (n, r)
    Lambda : np.ndarray, shape (r,)

    Returns
    -------
    x_next : np.ndarray, shape (n,)
        The predicted log-return after the last snapshot in X1.
    """
    x_last = X1[:, -1]  # shape (n,)
    c_last = np.linalg.pinv(Phi) @ x_last
    x_next = Phi @ (np.diag(Lambda) @ c_last)
    return x_next


##############################################################################
#                                MAIN BLOCK                                  #
##############################################################################
if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Example A: Rolling forecast with heatmap
    # ---------------------------------------------------------------------
    file_paths = ["cg_btc.csv", "cg_eth.csv", "cg_ltc.csv", "cg_xrp.csv"]

    m_values = range(5, 15)   # window sizes
    ell_values = range(1, 5)  # forecast horizons
    threshold = 0.5

    results, hot_spots = evaluate_hot_spots_rolling(
        file_paths, m_values, ell_values, threshold=threshold
    )

    print("\nAll Results (rolling forecast):")
    for (m, ell), rate in sorted(results.items()):
        print(f"(m={m}, ell={ell}): success rate = {rate:.2f}")

    print("\nHot Spots (threshold = 0.50):")
    for (m, ell), rate in sorted(hot_spots.items()):
        print(f"(m={m}, ell={ell}): success rate = {rate:.2f}")

    plot_hot_spots(results)

    # ---------------------------------------------------------------------
    # Example B: Single-step DMD usage example (with continuous-time eigenvalues)
    # ---------------------------------------------------------------------
    try:
        X1, X2, truncated_prices = load_and_prepare_data_single_step(file_paths)

        print("\n\n--- Single-Step DMD Usage Example ---")
        print(f"Shape of X1 (log returns): {X1.shape}")
        print(f"Shape of X2 (log returns): {X2.shape}")

        # Perform the single-step DMD
        Phi, Lambda, omega, b = dmd_decomposition_single_step(X1, X2)
        print("\nDMD Modes (Phi):")
        print(Phi)
        print("\nEigenvalues (Lambda):")
        print(Lambda)
        print("\nContinuous-time Eigenvalues (omega):")
        print(omega)
        print("\nMode Amplitudes (b) from first snapshot:")
        print(b)

        # Predict the next log return (one step beyond X1's last snapshot)
        next_log_return = predict_one_step_beyond_last_snapshot(X1, Phi, Lambda)
        print(f"\nPredicted Next State (log return) for each crypto:\n{next_log_return}")

        # Convert predicted log returns to predicted prices
        for i, prices in enumerate(truncated_prices):
            last_price = prices[-1]
            predicted_log_r = next_log_return[i]
            predicted_price = last_price * np.exp(predicted_log_r)

            print(f"\n--- Crypto {i+1} ---")
            print(f"Last actual price: {last_price:.4f}")
            print(f"Predicted log return: {predicted_log_r:.6f}")
            print(f"Predicted next price: {predicted_price:.4f}")

            # (Optional) Plot historical prices + predicted next price
            plt.figure(figsize=(10, 6))
            plt.plot(prices, label="Historical Prices", marker='o')
            plt.scatter(len(prices), predicted_price, color='red',
                        label="Predicted Next Price (1-step ahead)")

            plt.title(f"Crypto #{i+1}: Price Forecast")
            plt.xlabel("Days (aligned & truncated)")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.show()

    except ValueError as e:
        print(f"Error in single-step usage example: {e}")