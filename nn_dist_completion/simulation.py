"""
Functions for generating simulated data for distributional matrix completion experiments.
"""

import numpy as np
from itertools import product
from tqdm import tqdm
import scipy.stats

from .utils import normal_ppf, wasserstein2, empirical_quantile
from .nearest_neighbors import get_user_user_distances_row, estimate_row, search_eta


def simulate_nn_location_scale(M, N, n, n_runs, dist: str, seed=0):
    """
    Simulate nearest neighbors experiments with location-scale families.
    
    Parameters
    ----------
    M : int
        Number of rows
    N : int
        Number of columns
    n : int
        Number of samples per distribution
    n_runs : int
        Number of simulation runs
    dist : str
        Distribution type ('normal' or 'uniform')
    seed : int, optional
        Random seed (default: 0)
        
    Returns
    -------
    tuple
        (error_est, error_observed, num_neighbors) lists
    """
    np.random.seed(seed)
    error_est = []
    error_observed = []
    num_neighbors = []

    for _ in tqdm(range(n_runs), desc=f'{M}, {N}, {n}'):
        true_dists = {}

        # Table of n x m Gaussians with columns having same variance and rows having same means
        mean_rows = np.random.uniform(-5, 5, M)
        std_cols = np.random.uniform(0.1, 0.5, N)
        
        data_table = np.zeros((M, N, n))

        for i, j in product(range(M), range(N)):
            if dist == 'normal':
                data = np.sort(np.random.normal(mean_rows[i], std_cols[j], n))
                data_table[i, j, :] = data
                true_dists[i, j] = normal_ppf(mean_rows[i], std_cols[j])
            elif dist == 'uniform':
                data = np.sort(np.random.uniform(mean_rows[i], mean_rows[i] + std_cols[j], n))
                data_table[i, j, :] = data
                true_dists[i, j] = lambda q, mean=mean_rows[i], std=std_cols[j]: scipy.stats.uniform.ppf(q, mean, std)

        mask = np.ones((M, N)).astype(int)
        mask[0, 0] = 0
        
        eta = search_eta(data_table, mask)

        d = get_user_user_distances_row(data_table, mask, 0)
        est_dist = estimate_row(data_table, mask, 0, d, eta)
        
        error_est.append(wasserstein2(empirical_quantile(est_dist), true_dists[0, 0]))
        error_observed.append(wasserstein2(empirical_quantile(data_table[0, 0]), true_dists[0, 0]))

        num_neighbors.append(np.sum(d < eta))  # number of neighbors
    return error_est, error_observed, num_neighbors


def generate_normal_data(M, N, n, mean_range=(-5, 5), std_range=(1, 5), seed=None):
    """
    Generate a data table with normal distributions.
    
    Parameters
    ----------
    M : int
        Number of rows
    N : int
        Number of columns
    n : int
        Number of samples per distribution
    mean_range : tuple, optional
        Range for row means (default: (-5, 5))
    std_range : tuple, optional
        Range for column standard deviations (default: (1, 5))
    seed : int, optional
        Random seed
        
    Returns
    -------
    tuple
        (data_table, true_dists, mean_rows, std_cols)
    """
    if seed is not None:
        np.random.seed(seed)
    
    mean_rows = np.random.uniform(mean_range[0], mean_range[1], M)
    std_cols = np.random.uniform(std_range[0], std_range[1], N)
    
    data_table = np.zeros((M, N, n))
    true_dists = {}
    
    for i, j in product(range(M), range(N)):
        data = np.sort(np.random.normal(mean_rows[i], std_cols[j], n))
        data_table[i, j, :] = data
        true_dists[i, j] = scipy.stats.norm(mean_rows[i], std_cols[j])
    
    return data_table, true_dists, mean_rows, std_cols


def generate_uniform_data(M, N, n, mean_range=(-5, 5), std_range=(1, 5), seed=None):
    """
    Generate a data table with uniform distributions.
    
    Parameters
    ----------
    M : int
        Number of rows
    N : int
        Number of columns
    n : int
        Number of samples per distribution
    mean_range : tuple, optional
        Range for row means (default: (-5, 5))
    std_range : tuple, optional
        Range for column widths (default: (1, 5))
    seed : int, optional
        Random seed
        
    Returns
    -------
    tuple
        (data_table, true_dists, mean_rows, std_cols)
    """
    if seed is not None:
        np.random.seed(seed)
    
    mean_rows = np.random.uniform(mean_range[0], mean_range[1], M)
    std_cols = np.random.uniform(std_range[0], std_range[1], N)
    
    data_table = np.zeros((M, N, n))
    true_dists = {}
    
    for i, j in product(range(M), range(N)):
        data = np.sort(np.random.uniform(mean_rows[i], mean_rows[i] + std_cols[j], n))
        data_table[i, j, :] = data
        true_dists[i, j] = scipy.stats.uniform(mean_rows[i], std_cols[j])
    
    return data_table, true_dists, mean_rows, std_cols


def simulate_metrics_comparison(M, N, n, n_trials=60, alpha=0.05, seed=0):
    """
    Simulate comparison of Dist-NN vs baseline methods on various metrics.
    
    Parameters
    ----------
    M : int
        Number of rows
    N : int
        Number of columns
    n : int
        Number of samples per distribution
    n_trials : int, optional
        Number of trials (default: 60)
    alpha : float, optional
        Significance level for VaR (default: 0.05)
    seed : int, optional
        Random seed (default: 0)
        
    Returns
    -------
    dict
        Dictionary containing errors for different methods and metrics
    """
    np.random.seed(seed)

    # Dist-NN Errors
    std_dnn_errors = []
    var_dnn_errors = []
    mean_dnn_errors = []
    median_dnn_errors = []

    # Scalar NN Errors
    std_snn_errors = []
    var_snn_errors = []
    mean_snn_errors = []
    median_snn_errors = []

    # Random sample errors
    std_rand_errors = []
    var_rand_errors = []
    mean_rand_errors = []
    median_rand_errors = []

    from .utils import relative_error, empirical_quantile, normal_ppf
    from .nearest_neighbors import get_user_user_distances_row, estimate_row, search_eta

    for _ in tqdm(range(n_trials)):
        true_dists = {}

        # Table of n x m Gaussians with columns having same variance and rows having same means
        mean_rows = np.random.uniform(-5, 5, M)
        std_cols = np.random.uniform(1, 5, N)
        
        data_table = np.zeros((M, N, n))
        data_table_std = np.zeros((M, N, 1))
        data_table_var = np.zeros((M, N, 1))
        data_table_mean = np.zeros((M, N, 1))
        data_table_median = np.zeros((M, N, 1))

        for i, j in product(range(M), range(N)):
            data_unif = np.sort(np.random.uniform(0, 1, n))
            ppf = normal_ppf(mean_rows[i], std_cols[j])
            data = ppf(data_unif)
            
            data_table[i, j, :] = data
            true_dists[i, j] = ppf
            
            data_table_std[i, j, :] = np.std(data_table[i, j, :], keepdims=True)
            data_table_var[i, j, :] = empirical_quantile(-1 * data)(1 - alpha)
            data_table_mean[i, j, :] = np.mean(data_table[i, j, :], keepdims=True)
            data_table_median[i, j, :] = np.median(data_table[i, j, :], keepdims=True)

        mask = np.ones((M, N)).astype(int)
        mask[0, 0] = 0
        
        data_tables = [data_table, data_table_std, data_table_var, data_table_mean, data_table_median]
        ests = []
        
        for table in data_tables:
            eta = search_eta(table, mask)
            d = get_user_user_distances_row(table, mask, 0)
            est = estimate_row(table, mask, 0, d, eta)
            ests.append(est)

        var_nn = empirical_quantile(-1 * ests[0])(1 - alpha)
        var_random = empirical_quantile(-1 * data_table[0, 0])(1 - alpha)
        var_true = normal_ppf(-1 * mean_rows[0], std_cols[0])(1 - alpha)
        
        # Dist-NN
        std_dnn_errors.append(relative_error(np.std(ests[0]), std_cols[0]))
        var_dnn_errors.append(relative_error(var_nn, var_true))
        mean_dnn_errors.append(relative_error(np.mean(ests[0]), mean_rows[0]))
        median_dnn_errors.append(relative_error(np.median(ests[0]), mean_rows[0]))
        
        # Scalar-NN
        std_snn_errors.append(relative_error(ests[1][0], std_cols[0]))
        var_snn_errors.append(relative_error(ests[2][0], var_true))
        mean_snn_errors.append(relative_error(ests[3][0], mean_rows[0]))
        median_snn_errors.append(relative_error(ests[4][0], mean_rows[0]))
        
        # Random Sample
        std_rand_errors.append(relative_error(np.std(data_table[0, 0]), std_cols[0]))
        var_rand_errors.append(relative_error(var_random, var_true))
        mean_rand_errors.append(relative_error(np.mean(data_table[0, 0]), mean_rows[0]))
        median_rand_errors.append(relative_error(np.median(data_table[0, 0]), mean_rows[0]))
    
    return {
        'std_dnn': std_dnn_errors,
        'var_dnn': var_dnn_errors,
        'mean_dnn': mean_dnn_errors,
        'median_dnn': median_dnn_errors,
        'std_snn': std_snn_errors,
        'var_snn': var_snn_errors,
        'mean_snn': mean_snn_errors,
        'median_snn': median_snn_errors,
        'std_rand': std_rand_errors,
        'var_rand': var_rand_errors,
        'mean_rand': mean_rand_errors,
        'median_rand': median_rand_errors
    }

