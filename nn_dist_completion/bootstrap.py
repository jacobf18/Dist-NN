"""
Bootstrap methods for confidence intervals and variance estimation.
"""

import numpy as np
import scipy.stats


def get_variance_exact(x: float, num_samples: int, neighbor_true_dists: list) -> float:
    """
    Returns the confidence region for the true distribution
    given the true distributions of the neighbors.
    
    Parameters
    ----------
    x : float
        Quantile level (must be in [0, 1])
    num_samples : int
        Number of samples per distribution
    neighbor_true_dists : list
        List of true neighbor distributions (scipy.stats objects)
        
    Returns
    -------
    float
        Variance estimate
    """
    if x < 0 or x > 1:
        raise ValueError("x must be in the range [0,1]")
    m = len(neighbor_true_dists)
    brownian_bridge_var = x - (x**2)  # variance of the Brownian bridge
    neighbor_vars = np.sum(np.reciprocal([dist.pdf(dist.ppf(x)) ** 2 
                                          for dist in neighbor_true_dists]))  # variance of the neighbors
    finite_sample_var = brownian_bridge_var * neighbor_vars / (num_samples * (m**2))  # variance of the barycenter estimate
    return finite_sample_var


def get_variance_bootstrap_neighbors(x: float, 
                                     num_samples: int, 
                                     neighbor_true_dists: list,
                                     num_resamples_neighbors: int = 10) -> float:
    """
    Returns the confidence region for the true distribution
    given the true distributions of the neighbors using bootstrap.
    
    Parameters
    ----------
    x : float
        Quantile level (must be in [0, 1])
    num_samples : int
        Number of samples per distribution
    neighbor_true_dists : list
        List of true neighbor distributions (scipy.stats objects)
    num_resamples_neighbors : int, optional
        Number of bootstrap resamples (default: 10)
        
    Returns
    -------
    float
        Variance estimate
    """
    if x < 0 or x > 1:
        raise ValueError("x must be in the range [0,1]")
    variances = []
    m = len(neighbor_true_dists)
    neighbor_true_dists = np.array(neighbor_true_dists)
    
    for _ in range(num_resamples_neighbors):
        neighbors_inds = np.random.choice(m, m, replace=True)
        resampled_neighbors = neighbor_true_dists[neighbors_inds]
        var = get_variance_exact(x, num_samples, resampled_neighbors)
        variances.append(var)
    return np.mean(variances, axis=0)


def get_variance_bootstrap_samples(num_samples: int,
                                   neighbor_emp_dists: np.ndarray, 
                                   num_resamples_n: int = 10):
    """
    Bootstrap variance estimation by resampling samples.
    
    Parameters
    ----------
    num_samples : int
        Number of samples per distribution
    neighbor_emp_dists : np.ndarray
        Array of empirical neighbor distributions
    num_resamples_n : int, optional
        Number of bootstrap resamples (default: 10)
        
    Returns
    -------
    np.ndarray
        Variance estimates for each quantile level
    """
    m = len(neighbor_emp_dists)
    n = num_samples
    
    barycenters = []
    
    for _ in range(num_resamples_n):
        resampled_samples = np.zeros((m, n))
        for i in range(m):
            resampled_samples[i, :] = np.sort(np.random.choice(neighbor_emp_dists[i], n, replace=True))
        barycenter = np.mean(resampled_samples, axis=0)
        barycenters.append(barycenter)
    barycenters = np.array(barycenters)
    
    return np.var(barycenters, axis=0, ddof=1)


def get_variance_bootstrap_everything(
        num_samples: int,
        neighbor_emp_dists: np.ndarray, 
        num_resamples_n: int = 10, 
        num_resamples_neighbors: int = 10):
    """
    Returns the confidence region for the true distribution
    using the bootstrap method (resampling both neighbors and samples).
    
    Parameters
    ----------
    num_samples : int
        Number of samples per distribution
    neighbor_emp_dists : np.ndarray
        Array of empirical neighbor distributions
    num_resamples_n : int, optional
        Number of bootstrap resamples for samples (default: 10)
    num_resamples_neighbors : int, optional
        Number of bootstrap resamples for neighbors (default: 10)
        
    Returns
    -------
    np.ndarray
        Variance estimates for each quantile level
    """
    m = len(neighbor_emp_dists)
    n = num_samples
    
    barycenters = []
    for _ in range(num_resamples_neighbors):
        neighbors_inds = np.random.choice(m, m, replace=True)
        resampled_neighbors = neighbor_emp_dists[neighbors_inds]
        for _ in range(num_resamples_n):
            resampled_samples = np.zeros((m, n))
            for i in range(m):
                resampled_samples[i, :] = np.sort(np.random.choice(resampled_neighbors[i], n, replace=True))
            barycenter = np.mean(resampled_samples, axis=0)
            barycenters.append(barycenter)
    barycenters = np.array(barycenters)
    
    return np.var(barycenters, axis=0, ddof=1)

