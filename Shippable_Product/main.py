#Main file for rolling walk DMD program. Takes in stock data, does DMD, returns next predicted value (daily)
#and then "rolls" across entire dataset to create heatman of m and l values and shows their successes (avg)
import numpy as np
import matplotlib.pyplot as plt

import rolling_heatmap
import data_preparation
import single_step_forecast
import dmd_functions

if __name__ == "__main__":

    #roll over the entire data set, and then average the success in a heat map for different values of
    # l and m

    file_paths = ["cg_btc.csv", "cg_eth.csv", "cg_ltc.csv", "cg_xrp.csv", "cg_bnb.csv", "cg_bch.csv", "cg_link.csv", "cg_etc.csv"]

    m_values = range(1, 20)
    ell_values = range(1, 9)
    threshold = 0.5

    results, hot_spots = rolling_heatmap.evaluate_hot_spots_rolling(
        file_paths, m_values, ell_values, threshold=threshold
    )


    #chatgpt:
    print("\nAll Results (rolling forecast):")
    for (m, ell), rate in sorted(results.items()):
        print(f"(m={m}, ell={ell}): success rate = {rate:.2f}")

    print("\nHot Spots (threshold = 0.50):")
    for (m, ell), rate in sorted(hot_spots.items()):
        print(f"(m={m}, ell={ell}): success rate = {rate:.2f}")

    rolling_heatmap.plot_hot_spots(results)


    #single step DMD with continuous time eigenvalues


    try:
        #load data from csv files:
        X1, X2, truncated_prices = data_preparation.load_and_prepare_data_single_step(file_paths)

        #show x1 and x2 for debug
        print("\n\n--- Single-Step DMD Usage Example ---")
        print(f"Shape of X1 (log returns): {X1.shape}")
        print(f"Shape of X2 (log returns): {X2.shape}")

        # Perform the single-step DMD
        Phi, Lambda, omega, b = dmd_functions.dmd_decomposition_single_step(X1, X2)
        print("\nDMD Modes (Phi):")
        print(Phi)
        print("\nEigenvalues (Lambda):")
        print(Lambda)
        print("\nContinuous-time Eigenvalues (omega):")
        print(omega)
        print("\nMode Amplitudes (b) from first snapshot:")
        print(b)

         # Predict the next LOG return (one step beyond X1's last snapshot)
        next_log_return = single_step_forecast.predict_one_step_beyond_last_snapshot(X1, Phi, Lambda)
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

            #plot historical prices and the next PREDICTED price
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