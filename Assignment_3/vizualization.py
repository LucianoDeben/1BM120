import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy


def moving_average(values, window):
    """
    Compute the moving average of a numpy array.
    Efficiently handles edge cases and improves performance.
    """
    if len(values) < window or window <= 0:
        raise ValueError("Window size must be positive and less than or equal to the length of the data.")
    return np.convolve(values, np.ones(window)/window, 'valid')

def plot_results(log_folder, title='Training Reward', window=50, color='blue', label='Reward', save_fig=False, fig_path='reward_plot.png'):
    """
    Plots the average accumulated reward from the log folder with a moving average smoothing.
    Allows customization of the plot and the option to save to a file.
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    if len(x) == 0 or len(y) == 0:
        raise ValueError("The log folder does not contain valid data.")
    cumulative_rewards = np.cumsum(y)
    average_accumulated_rewards = cumulative_rewards / (np.arange(len(cumulative_rewards)) + 1)
    y_smooth = moving_average(average_accumulated_rewards, window=window)
    x_smooth = x[:len(y_smooth)]
    
    plt.figure(figsize=(16, 8))
    plt.plot(x_smooth, y_smooth, color=color, label=label)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Average Accumulated Reward')
    plt.title(title)
    plt.legend()
    if save_fig:
        plt.savefig(fig_path)
    else:
        plt.show()