import numpy as np
import pandas as pd

def load_log_return_matrix(file_paths, how="inner", fill=None):
    import pandas as pd, numpy as np

    # Load each series as daily mean without dropping NaNs yet
    daily_series = []
    for p in file_paths:
        df = pd.read_csv(p)
        if {'snapped_at','price'} - set(df.columns):
            raise ValueError(f"{p} missing 'snapped_at' or 'price'")
        s = (df.assign(snapped_at=pd.to_datetime(df['snapped_at']))
                .sort_values('snapped_at')
                .set_index('snapped_at')['price']
                .resample('D').mean())
        daily_series.append(s.rename(p))

    # Combine into one DataFrame with a common date axis
    prices = pd.concat(daily_series, axis=1, join='outer')
    if fill == "ffill":
        prices = prices.ffill()
    elif fill == "bfill":
        prices = prices.bfill()

    # Choose alignment policy
    if how == "inner":
        prices = prices.dropna(how='any')
    elif how == "outer":
        pass  # keep NaNs if you plan to impute differently
    else:
        raise ValueError("how must be 'inner' or 'outer'")

    # Compute log-returns on the aligned grid
    logret      = np.log(prices/prices.shift(1)).dropna(how='any')
    data_matrix = logret.to_numpy().T  # (n, T)
    return data_matrix

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
    all_data         = []
    original_prices  = []
    start_dates      = []

    # 1) First pass: read each file, store earliest date among all cryptos
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        if 'snapped_at' not in df.columns or 'price' not in df.columns:
            raise ValueError(f"File {file_path} missing 'snapped_at' or 'price' columns.")

        df['snapped_at']  = pd.to_datetime(df['snapped_at'])
        df                = df.sort_values(by='snapped_at')
        df_daily          = df.set_index('snapped_at').resample('D').mean()

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
    aligned_data    = []
    aligned_prices  = []
    for file_path in file_paths:
        df                = pd.read_csv(file_path)
        df['snapped_at']  = pd.to_datetime(df['snapped_at'])
        df                = df.sort_values(by='snapped_at')
        df_daily          = df.set_index('snapped_at').resample('D').mean()
        df_daily          = df_daily[df_daily.index >= common_start_date]
        prices            = df_daily['price'].dropna().values

        if len(prices) > 1:
            log_returns = np.log(prices[1:] / prices[:-1])
        else:
            log_returns = np.array([])

        aligned_data.append(log_returns)
        aligned_prices.append(prices)

    # 4) Truncate all log-return series to the same minimum length
    min_length        = min(len(lr) for lr in aligned_data if len(lr) > 0)
    truncated_data    = [lr[:min_length] for lr in aligned_data]
    truncated_prices  = [p[:min_length + 1] for p in aligned_prices]

    # 5) Stack into data_matrix
    data_matrix = np.array(truncated_data)  # shape (n, T)

    # 6) Build X1, X2 for single-step shift
    X1 = data_matrix[:, :-1]  # shape (n, T-1)
    X2 = data_matrix[:, 1:]   # shape (n, T-1)

    print("Debug:")
    print(X1)
    print(X2)
    input()

    return X1, X2, truncated_prices