"""
Nearest neighbors algorithms for distributional matrix completion.
"""

import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from copy import deepcopy
from hyperopt import hp, tpe, fmin
from tqdm import tqdm

from .utils import emp_wasserstein2, dissim, barycenter


def get_user_user_distances_fast(data_table, mask):
    """
    Vectorized version of user-user distances.
    Assumes data_table is a 3D numpy array.
    Assumes mask is a 2D numpy array.
    
    Assumes data is already sorted.
    
    Parameters
    ----------
    data_table : np.ndarray
        3D array of shape (n_rows, n_cols, n_samples)
    mask : np.ndarray
        2D array indicating observed entries
        
    Returns
    -------
    np.ndarray
        2D array of pairwise distances between rows
    """
    data_table = np.copy(data_table)
    mask = np.copy(mask)
    
    data_table[mask == 0, :] = np.inf  # mask values as infinity for differences
    dists = np.power(data_table[:, None, :, :] - data_table, 2)  # each pair of rows differenced
    row_dists = np.mean(dists, axis=3)  # average squared differences across samples
    np.nan_to_num(row_dists, copy=False, nan=0, posinf=0)  # convert nan and inf values to 0
    
    overlap = np.sum(mask[:, None, :] * mask, axis=2)  # get number of overlap elements between each pair of users
    row_dists = np.sum(row_dists, axis=2) / overlap  # average dists across columns
    
    np.fill_diagonal(row_dists, np.inf)  # each user is not a nearest neighbor of itself

    return row_dists


def estimate_fast(data_table, mask, row, col, row_dists, eta):
    """
    Vectorized version of nearest neighbors.
    Assumes data_table is a 3D numpy array.
    Assumes mask is a 2D numpy array.
    
    Assumes data is already sorted.
    
    Parameters
    ----------
    data_table : np.ndarray
        3D array of shape (n_rows, n_cols, n_samples)
    mask : np.ndarray
        2D array indicating observed entries
    row : int
        Target row index
    col : int
        Target column index
    row_dists : np.ndarray
        Pairwise distances between rows
    eta : float
        Distance threshold
        
    Returns
    -------
    np.ndarray
        Estimated distribution for (row, col)
    """
    NN = row_dists < eta  # threshold to get nearest neighbors
    NN = NN[row]  # get the nearest neighbors for the target user
    if np.sum(NN) > 0:
        return np.mean(data_table[NN, col, :], axis=0)  # barycenter of nearest neighbors
    else:
        data_masked = data_table * mask[:, :, None]  # mask the data to exclude data not observed
        return np.mean(data_masked[:, col, :], axis=0)  # barycenter of entire observed column


def get_user_user_distances(data_table, mask, n_rows, n_cols):
    """
    Compute pairwise distances between rows (non-vectorized version).
    
    Parameters
    ----------
    data_table : dict or np.ndarray
        Data table structure
    mask : np.ndarray
        2D array indicating observed entries
    n_rows : int
        Number of rows
    n_cols : int
        Number of columns
        
    Returns
    -------
    dict
        Dictionary of pairwise distances
    """
    d = {}
    
    for i in range(n_rows):
        for u in range(i, n_rows):
            overlap = {j for j in range(n_cols) if mask[i, j] == 1 and mask[u, j] == 1}
            # if no overlap or looking at the same user
            if i == u or len(overlap) == 0:
                d[i, u] = np.inf
                d[u, i] = np.inf
                continue
            # if overlap, then calculate dissimilarity
            d[i, u] = dissim([data_table[i, j] for j in range(n_cols) if j in overlap],
                            [data_table[u, j] for j in range(n_cols) if j in overlap])
            d[u, i] = d[i, u]
    return d


def estimate(data_table, mask, row, col, eta, n_rows, n_cols, distances):
    """
    Estimate distribution for a missing entry (non-vectorized version).
    
    Parameters
    ----------
    data_table : dict or np.ndarray
        Data table structure
    mask : np.ndarray
        2D array indicating observed entries
    row : int
        Target row index
    col : int
        Target column index
    eta : float
        Distance threshold
    n_rows : int
        Number of rows
    n_cols : int
        Number of columns
    distances : dict
        Dictionary of pairwise distances
        
    Returns
    -------
    np.ndarray
        Estimated distribution
    """
    NN = {k for k in range(n_rows) if distances[row, k] <= eta}
    avg_inds = {}
    if len(NN) > 0:
        avg_inds = NN
    else:
        avg_inds = {k for k in range(n_rows) if mask[k, col] == 1}
    return barycenter([data_table[k, col] for k in avg_inds])


def get_user_user_distances_row(data_table, mask, row):
    """
    Compute distances from a specific row to all other rows.
    
    Parameters
    ----------
    data_table : np.ndarray
        3D array of shape (n_rows, n_cols, n_samples)
    mask : np.ndarray
        2D array indicating observed entries
    row : int
        Target row index
        
    Returns
    -------
    np.ndarray
        1D array of distances from target row to all rows
    """
    rows, cols, samples = data_table.shape
    dists = np.zeros(rows)
    for i in range(rows):
        if i == row:
            dists[i] = np.inf
            continue
        dist = 0
        count = 0
        for j in range(cols):
            if mask[row, j] == 1 and mask[i, j] == 1:
                dist += emp_wasserstein2(data_table[row, j, :], data_table[i, j, :])
                count += 1
        if count > 0:
            dists[i] = dist / count
        else:
            dists[i] = np.inf
    return dists


def estimate_row(data_table, mask, col, row_dists, eta):
    """
    Estimate distribution for a missing entry in a specific row.
    
    Parameters
    ----------
    data_table : np.ndarray
        3D array of shape (n_rows, n_cols, n_samples)
    mask : np.ndarray
        2D array indicating observed entries
    col : int
        Target column index
    row_dists : np.ndarray
        1D array of distances from target row to all rows
    eta : float
        Distance threshold
        
    Returns
    -------
    np.ndarray
        Estimated distribution
    """
    NN = row_dists < eta  # threshold to get nearest neighbors
    
    if np.sum(NN) > 0:
        masked = (NN * mask[:, col]) == 1  # need the ==1 to convert back to booleans
        return np.mean(data_table[masked, col, :], axis=0)
    else:
        masked = (mask[:, col]) == 1
        return np.mean(data_table[masked, col, :], axis=0)


def evaluate_eta(data_table, mask, k, eta):
    """
    Run cross validation on the dataset with a given eta.
    
    Parameters
    ----------
    data_table : np.ndarray
        3D array of shape (n_rows, n_cols, n_samples)
    mask : np.ndarray
        2D array indicating observed entries
    k : int
        Number of folds for cross-validation
    eta : float
        Distance threshold
        
    Returns
    -------
    float
        Average cross-validation error
    """
    n_rows, n_cols, _ = data_table.shape
    observed_inds = np.array([(i, j) for i, j in product(range(n_rows), range(n_cols)) if mask[i, j] == 1])
    np.random.shuffle(observed_inds)
    chunks = np.array_split(observed_inds, k)
    error = 0
    
    for k_ind in range(k):
        test = chunks[k_ind]  # left out
        train = np.concatenate([chunks[i] for i in range(k) if i != k_ind])  # everything else
        test_set = {tuple(t) for t in test}
        
        new_mask = np.copy(mask)
        for i, j in product(range(n_rows), range(n_cols)):
            if (i, j) in test_set:
                new_mask[i, j] = 0
        dists = get_user_user_distances_fast(data_table, new_mask)
        
        for i, j in product(range(n_rows), range(n_cols)):
            if (i, j) in test_set:
                est = estimate_fast(data_table, new_mask, i, j, dists, eta)
                error += emp_wasserstein2(est, data_table[i, j])
                
    error /= len(observed_inds)
    return error


def evalute_eta_row(data_table, row, mask, eta):
    """
    Cross validate just on the row we are interested in estimating.
    
    Parameters
    ----------
    data_table : np.ndarray
        3D array of shape (n_rows, n_cols, n_samples)
    row : int
        Target row index
    mask : np.ndarray
        2D array indicating observed entries
    eta : float
        Distance threshold
        
    Returns
    -------
    float
        Average cross-validation error for the row
    """
    n_rows, n_cols, _ = data_table.shape
    observed_inds = np.array([j for j in range(n_cols) if mask[row, j] == 1])
    error = 0
    for k in observed_inds:
        new_mask = np.copy(mask)
        new_mask[row, k] = 0
        dists = get_user_user_distances_row(data_table, new_mask, row)
        est = estimate_row(data_table, new_mask, k, dists, eta)
        error += emp_wasserstein2(est, data_table[row, k])
    return error / len(observed_inds)


def search_eta(data_table, mask, row=0):
    """
    Search for an optimal eta using cross validation on
    the observed data.
    
    Parameters
    ----------
    data_table : np.ndarray
        3D array of shape (n_rows, n_cols, n_samples)
    mask : np.ndarray
        2D array indicating observed entries
    row : int, optional
        Row to optimize for (default: 0)
        
    Returns
    -------
    float
        Optimal eta value
    """
    def obj(eta):
        return evalute_eta_row(data_table, row, mask, eta)
    
    best_eta = fmin(fn=obj, verbose=False, space=hp.uniform('eta', 0.0001, 10.0), 
                    algo=tpe.suggest, max_evals=30)
    return best_eta['eta']

