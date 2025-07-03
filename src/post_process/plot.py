import argparse
import os

import numpy as np

from src.utils.util import plot_training_progress


def apply_moving_average(metrics_history, window_size=10):
    """Applies a moving average filter to each metric in the history."""
    smoothed_metrics = {}
    for key, values in metrics_history.items():
        values = np.array(values)
        if window_size <= 1 or len(values) < window_size:
            smoothed_metrics[key] = values
            continue

        # Calculate the moving average using convolution
        # 'valid' mode ensures that the window is always full
        smoothed_values = np.convolve(values, np.ones(window_size), "valid") / window_size

        # To align the smoothed data with the original, we can pad the start
        # This will make the smoothed value at index `i` be the average of `i-window_size+1` to `i`
        # The first few values will not have a full window average. We can just repeat the first smoothed value.
        padded_smoothed_values = np.pad(smoothed_values, (window_size - 1, 0), "edge")

        smoothed_metrics[key] = padded_smoothed_values
    return smoothed_metrics


def plot_episode_final_results(npz_file_path, save_dir=None, window_size=1):
    """
    Loads metrics from a .npz file, extracts the metrics from the final
    step of each episode, and generates a plot.
    """
    if not os.path.exists(npz_file_path):
        print(f"Error: Metrics file not found at {npz_file_path}")
        return

    if save_dir is None:
        save_dir = os.path.dirname(npz_file_path)
        os.makedirs(save_dir, exist_ok=True)

    data = np.load(npz_file_path)

    required_keys = ["episode_numbers", "rewards", "max_displacements", "max_stresses", "model_volumes"]
    if not all(key in data for key in required_keys):
        print(f"Error: The npz file is missing one of the required keys: {required_keys}")
        return

    episode_numbers = data["episode_numbers"]

    if len(episode_numbers) == 0:
        print("Metrics file is empty. Nothing to plot.")
        return

    rewards = data["rewards"]
    max_displacements = data["max_displacements"]
    max_stresses = data["max_stresses"]
    model_volumes = data["model_volumes"]

    change_indices = np.where(episode_numbers[:-1] != episode_numbers[1:])[0]
    last_step_indices = np.concatenate((change_indices, [len(episode_numbers) - 1]))

    final_metrics = {
        "rewards": rewards[last_step_indices],
        "max_displacements": max_displacements[last_step_indices],
        "max_stresses": max_stresses[last_step_indices],
        "model_volumes": model_volumes[last_step_indices],
    }

    if window_size > 1:
        metrics_to_plot = apply_moving_average(final_metrics, window_size)
        output_filename = f"metrics_by_episode_smoothed_ws{window_size}.png"
    else:
        metrics_to_plot = final_metrics
        output_filename = "metrics_by_episode.png"

    plot_training_progress(metrics_to_plot, save_dir, filename=output_filename)
    print(f"Final episode metrics plot saved to {os.path.join(save_dir, output_filename)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process and plot training results.")
    parser.add_argument("--metrics_path", type=str, required=True, help="Path to the metrics.npz file.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the plot.")
    parser.add_argument(
        "--window_size",
        type=int,
        default=1,
        help="Window size for moving average. Default is 1 (no smoothing).",
    )
    args = parser.parse_args()

    plot_episode_final_results(args.metrics_path, args.save_dir, args.window_size)
