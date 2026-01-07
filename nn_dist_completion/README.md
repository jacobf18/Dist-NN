# Nearest Neighbors Distribution Completion

This package provides tools for distributional matrix completion using Wasserstein nearest neighbors.

## Installation

Install the package using pip:

```bash
pip install -e .
```

Or install in development mode with optional dependencies:

```bash
pip install -e ".[dev]"
```

## Package Structure

- `utils.py`: Utility functions for Wasserstein distance calculations and quantile functions
- `nearest_neighbors.py`: Core nearest neighbors algorithms for distributional matrix completion
- `simulation.py`: Functions for generating simulated data
- `plotting.py`: Visualization and plotting functions
- `bootstrap.py`: Bootstrap methods for confidence intervals
- `config.py`: Configuration settings for matplotlib

## Quick Start

```python
import numpy as np
from nn_dist_completion import simulation, nearest_neighbors, plotting, config

# Setup plotting configuration
config.setup_plotting()

# Generate simulated data
M, N, n = 100, 20, 30
data_table, true_dists, mean_rows, std_cols = simulation.generate_normal_data(M, N, n)

# Create a mask (missing entry at (0, 0))
mask = np.ones((M, N)).astype(int)
mask[0, 0] = 0

# Find optimal eta using cross-validation
eta = nearest_neighbors.search_eta(data_table, mask)

# Compute distances and estimate missing entry
d = nearest_neighbors.get_user_user_distances_row(data_table, mask, 0)
est_dist = nearest_neighbors.estimate_row(data_table, mask, 0, d, eta)

# Plot results
plotting.plot_cdf_comparison(
    est_dist, 
    data_table[0, 0], 
    true_dists[0, 0].ppf,
    dist_type='normal',
    save_path='figures/example.pdf'
)
```

## Running Experiments

### Error Rate Experiments

```python
from nn_dist_completion import simulation

# Run simulation for different parameter combinations
errors_gaussian = {}
rows = [10, 50, 100, 1000]
cols = [30]
samples = [2, 10, 100, 500, 1000]
n_runs = 10

for M, N, n in product(rows, cols, samples):
    errors_gaussian[(M, N, n)] = simulation.simulate_nn_location_scale(
        M, N, n, n_runs, 'normal'
    )
```

### Metrics Comparison

```python
from nn_dist_completion import simulation

# Compare Dist-NN vs baseline on various metrics
dist_errors_dict = simulation.simulate_metrics_comparison(
    M=100, N=20, n=10, n_trials=60
)
```

## Module Documentation

### utils.py

Core utility functions:
- `emp_wasserstein2()`: Compute squared 2-Wasserstein distance between empirical distributions
- `wasserstein2()`: Compute squared 2-Wasserstein distance between distributions
- `empirical_quantile()`: Create empirical quantile function
- `normal_ppf()`, `uniform_ppf()`: Create quantile functions for distributions

### nearest_neighbors.py

Main algorithms:
- `get_user_user_distances_fast()`: Vectorized computation of pairwise distances
- `get_user_user_distances_row()`: Compute distances from a specific row
- `estimate_row()`: Estimate missing distribution using nearest neighbors
- `search_eta()`: Find optimal distance threshold using cross-validation

### simulation.py

Data generation:
- `generate_normal_data()`: Generate data table with normal distributions
- `generate_uniform_data()`: Generate data table with uniform distributions
- `simulate_nn_location_scale()`: Run full simulation experiments
- `simulate_metrics_comparison()`: Compare methods on various metrics

### plotting.py

Visualization functions:
- `plot_setup_visualization()`: Create setup visualization
- `plot_cdf_comparison()`: Compare estimated vs true CDFs
- `plot_error_vs_samples()`: Plot error vs number of samples
- `plot_error_vs_rows()`: Plot error vs number of rows
- `plot_confidence_bands()`: Plot confidence bands for estimates
- `plot_metrics_comparison()`: Compare methods across metrics

### bootstrap.py

Bootstrap methods:
- `get_variance_exact()`: Exact variance calculation
- `get_variance_bootstrap_neighbors()`: Bootstrap variance with neighbor resampling
- `get_variance_bootstrap_samples()`: Bootstrap variance with sample resampling
- `get_variance_bootstrap_everything()`: Full bootstrap (both neighbors and samples)
