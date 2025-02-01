import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Assuming these modules are the same as in your original code
import rolling_heatmap
import data_preparation
import single_step_forecast
import dmd_functions
import slider_heatmap

def evaluate_rolling_chunks(
    data_matrix,
    m_values,
    ell_values,
    chunk_size=100,
    stride=10,
    threshold=0.5
):
    """
    Splits data_matrix (n, T) into multiple chunks along the time dimension.
    For each chunk, computes rolling forecast success rates for each (m, ell).

    Parameters
    ----------
    data_matrix : np.ndarray, shape (n, T)
        Full log-return dataset for all cryptos.
    m_values : iterable
        Window sizes to test.
    ell_values : iterable
        Forecast horizons to test.
    chunk_size : int
        Length (in days) of each time chunk.
    stride : int
        How many days we move forward between consecutive chunks.
    threshold : float, optional
        For identifying hot spots above a certain success rate.

    Returns
    -------
    chunk_results : list of dict
        Each element is a dict with keys:
          {
            'start': (int) index of chunk start in data_matrix,
            'end': (int) index of chunk end in data_matrix,
            'results': dict of {(m, ell): success_rate},
            'hot_spots': dict of {(m, ell): success_rate},
          }
    """
    n, T = data_matrix.shape
    chunk_results = []

    start = 0
    while start + chunk_size <= T:
        end = start + chunk_size
        segment_matrix = data_matrix[:, start:end]

        # Evaluate success rates on just this segment
        results = {}
        for m in m_values:
            for ell in ell_values:
                successes = 0.0
                total_checks = 0

                # Rolling within the chunk
                seg_len = segment_matrix.shape[1]
                for t in range(seg_len - m - ell + 1):
                    X1 = segment_matrix[:, t : t + m - 1]
                    X2 = segment_matrix[:, t + 1 : t + m]

                    if X1.shape[1] < 1:
                        continue

                    try:
                        Phi, Lambda, b_last = dmd_functions.dmd_decomposition_rolling(X1, X2)
                        Lambda_ell = Lambda ** ell
                        predicted_log_return = (Phi @ (Lambda_ell * b_last)).real

                        actual_day = t + m - 1 + ell
                        if actual_day < seg_len:
                            actual_log_return = segment_matrix[:, actual_day]
                            sr = rolling_heatmap.evaluate_directional_success(
                                actual_log_return, predicted_log_return
                            )
                            successes += sr * segment_matrix.shape[0]
                            total_checks += segment_matrix.shape[0]

                    except np.linalg.LinAlgError:
                        pass

                if total_checks > 0:
                    overall_success_rate = successes / total_checks
                    results[(m, ell)] = overall_success_rate

        hot_spots = {k: v for k, v in results.items() if v >= threshold}

        chunk_results.append({
            'start': start,
            'end': end,
            'results': results,
            'hot_spots': hot_spots
        })

        start += stride

    return chunk_results

def build_heatmap_array(results):
    """
    Convert results dict {(m, ell): rate} into a 2D numpy array
    for plotting via imshow.

    Returns
    -------
    heatmap : 2D np.ndarray
    m_vals_sorted : list
    ell_vals_sorted : list
    """
    if not results:
        return np.zeros((1, 1)), [], []

    m_values = sorted(set(k[0] for k in results.keys()))
    ell_values = sorted(set(k[1] for k in results.keys()))

    heatmap = np.zeros((len(m_values), len(ell_values)))
    for (m, ell), rate in results.items():
        i = m_values.index(m)
        j = ell_values.index(ell)
        heatmap[i, j] = rate

    return heatmap, m_values, ell_values

def timelapse_heatmap(file_paths, m_values, ell_values,
                      chunk_size=100, stride=10, threshold=0.5):
    # 1) Load entire dataset
    data_matrix = data_preparation.load_log_return_matrix(file_paths)

    # 2) Evaluate chunks
    chunk_results = evaluate_rolling_chunks(
        data_matrix,
        m_values,
        ell_values,
        chunk_size=chunk_size,
        stride=stride,
        threshold=threshold
    )
    if not chunk_results:
        print("No chunks produced.")
        return

    # 3) Convert chunk results into heatmaps
    heatmaps = []
    for cr in chunk_results:
        hm, _, _ = build_heatmap_array(cr['results'])
        heatmaps.append(hm)

    # Compute global color range across all heatmaps for consistency
    global_min = min(hm.min() for hm in heatmaps)
    global_max = max(hm.max() for hm in heatmaps)

    # 4) Set up plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2)

    # Show the first chunk’s heatmap
    im = ax.imshow(
        heatmaps[0],
        cmap='hot',
        interpolation='nearest',
        aspect='auto',
        vmin=global_min,
        vmax=global_max
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Success Rate")

    ax.set_title("Timelapse Heatmap (chunk 0)")

    # (Optional) If you have consistent m_values/ell_values across all chunks,
    # you can set ticks here:
    # ax.set_xticks(...)
    # ax.set_yticks(...)

    # -- Add a text box to display the average rate --
    # Using normalized axes coordinates, so (0.05, 0.9) is near top-left inside the plot.
    text_box = ax.text(
        0.05, 0.90, "",
        transform=ax.transAxes,
        ha="left", va="center",
        color="white", weight="bold",
        bbox=dict(facecolor="black", alpha=0.3, edgecolor="none")
    )

    # Initialize with the average of the first heatmap
    avg_initial = heatmaps[0].mean()
    text_box.set_text(f"Avg: {avg_initial:.3f}")

    # 5) Slider
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Chunk Index',
        valmin=0,
        valmax=len(chunk_results) - 1,
        valinit=0,
        valstep=1,
        orientation='horizontal'
    )

    # 6) Update function
    def update(val):
        idx = int(slider.val)
        im.set_data(heatmaps[idx])

        # Keep global color scale
        # im.set_clim(global_min, global_max)  # Not needed if already set above

        # Update title
        start_i = chunk_results[idx]['start']
        end_i = chunk_results[idx]['end']
        ax.set_title(f"Timelapse Heatmap (chunk {idx}): Days [{start_i}:{end_i}]")

        # Update the average text
        avg_rate = heatmaps[idx].mean()
        text_box.set_text(f"Avg: {avg_rate:.3f}")

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

    # 6) Update function for slider
    def update(val):
        idx = int(slider.val)
        im.set_data(heatmaps[idx])

        # Force color scale to the range of the new chunk
        # (If you'd rather keep a global scale, comment these out)
        im.set_clim(vmin=heatmaps[idx].min(), vmax=heatmaps[idx].max())

        # Update title
        start_i = chunk_results[idx]['start']
        end_i = chunk_results[idx]['end']
        ax.set_title(f"Chunk {idx}: Days [{start_i}:{end_i}]")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # 7) Show the interactive plot
    plt.show()