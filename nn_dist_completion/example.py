"""
Example script demonstrating how to use the nn_dist_completion package.
"""

import numpy as np
from itertools import product
from nn_dist_completion import (
    simulation, 
    nearest_neighbors, 
    plotting, 
    config,
    utils
)

# Setup plotting configuration
config.setup_plotting()

# Example 1: Generate data and estimate missing entry
print("Example 1: Basic usage")
M, N, n = 100, 20, 30
data_table, true_dists, mean_rows, std_cols = simulation.generate_normal_data(
    M, N, n, seed=42
)

# Create a mask (missing entry at (0, 0))
mask = np.ones((M, N)).astype(int)
mask[0, 0] = 0

# Find optimal eta using cross-validation
print("Searching for optimal eta...")
eta = nearest_neighbors.search_eta(data_table, mask)

# Compute distances and estimate missing entry
d = nearest_neighbors.get_user_user_distances_row(data_table, mask, 0)
est_dist = nearest_neighbors.estimate_row(data_table, mask, 0, d, eta)

# Calculate errors
est_error = utils.wasserstein2(
    utils.empirical_quantile(est_dist), 
    true_dists[0, 0].ppf
)
obs_error = utils.wasserstein2(
    utils.empirical_quantile(data_table[0, 0]), 
    true_dists[0, 0].ppf
)

print(f"Estimation error: {est_error:.6f}")
print(f"Observed error: {obs_error:.6f}")
print(f"Number of neighbors: {np.sum(d < eta)}")
print()

# Example 2: Run a small simulation
print("Example 2: Running simulation")
errors_est, errors_obs, num_neighbors = simulation.simulate_nn_location_scale(
    M=10, N=10, n=30, n_runs=5, dist='normal'
)
print(f"Average estimation error: {np.mean(errors_est):.6f}")
print(f"Average observed error: {np.mean(errors_obs):.6f}")
print(f"Average number of neighbors: {np.mean(num_neighbors):.2f}")
print()

# Example 3: Plot CDF comparison
print("Example 3: Creating CDF comparison plot")
plotting.plot_cdf_comparison(
    est_dist, 
    data_table[0, 0], 
    true_dists[0, 0].ppf,
    n_samples=2,
    dist_type='normal',
    save_path='figures/example_cdf.pdf'
)
print("Plot saved to figures/example_cdf.pdf")

