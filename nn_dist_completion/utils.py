"""
Utility functions for Wasserstein distance calculations and quantile functions.
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, uniform
from numpy import quantile


def emp_wasserstein2(u: np.array, v: np.array):
    """
    Returns the squared 2-Wasserstein distance between two
    empirical distributions represented by arrays.
    Assumes the arrays of equal size and are sorted.
    
    Parameters
    ----------
    u : np.array
        First sorted array
    v : np.array
        Second sorted array
        
    Returns
    -------
    float
        Squared 2-Wasserstein distance
    """
    u_sorted = u
    v_sorted = v
    
    return np.sum(np.power(u_sorted - v_sorted, 2)) / len(u)


def dissim(list1: list[np.array], list2: list[np.array]) -> float:
    """
    Returns a dissimilarity measure between two lists of
    data arrays. Only measures dissimilarity between
    lists that are observed.
    
    If no observations overlap then returns infinity.
    Else returns the average 2-wasserstein distance.
    
    Parameters
    ----------
    list1 : list[np.array]
        First list of arrays
    list2 : list[np.array]
        Second list of arrays
        
    Returns
    -------
    float
        Average Wasserstein distance or infinity if no overlap
    """
    size = 0
    distance = 0
    for ind in range(len(list1)):
        if len(list1[ind]) == 1 and len(list2[ind]):
            distance += (list1[ind][0] - list2[ind][0]) ** 2
        else:
            distance += emp_wasserstein2(list1[ind], list2[ind])
        size += 1
    if size == 0:
        return float('inf')
    return distance / size


def get_col(table, col):
    """
    Utility function to get the column of a table
    as a 1-dim list.
    
    Parameters
    ----------
    table : list
        Table structure
    col : int
        Column index
        
    Returns
    -------
    list
        Column values
    """
    return [table[i][col] for i in range(len(table))]


def barycenter(lists: list[np.array]):
    """
    Returns the barycenter of a list of empirical distributions
    with the same number of samples.
    
    This is just the average of the order statistics of
    each empirical distribution.
    
    Assumes the arrays are of equal length and sorted.
    
    Parameters
    ----------
    lists : list[np.array]
        List of sorted arrays
        
    Returns
    -------
    np.array
        Barycenter (average) of the distributions
    """
    sorted_data = lists
    
    sum_data = 0
    for l in sorted_data:
        sum_data += l
    
    return sum_data / len(lists)


def wasserstein2(inv_cdf1, inv_cdf2):
    """
    Compute the squared 2-Wasserstein metric.
    
    Parameters
    ----------
    inv_cdf1 : callable
        First inverse CDF function
    inv_cdf2 : callable
        Second inverse CDF function
        
    Returns
    -------
    float
        Squared 2-Wasserstein distance
    """
    # Combine the inner part of the integral into one function
    # Estimate integral via quadrature
    y, _ = quad(lambda x: np.power((inv_cdf1(x) - inv_cdf2(x)), 2), 0, 1)
    return y


def wasserstein2_mc(ppf1, ppf2):
    """
    Compute squared 2-Wasserstein metric using Monte Carlo.
    
    Parameters
    ----------
    ppf1 : callable
        First quantile function
    ppf2 : callable
        Second quantile function
        
    Returns
    -------
    float
        Squared 2-Wasserstein distance
    """
    N = 10000
    u = np.random.uniform(0, 1, N)
    return np.sum(np.power(ppf1(u) - ppf2(u), 2)) / N


def empirical_quantile(data_arr):
    """
    Returns a function handle for the empirical quantile
    function given a 1-d dataset.
    
    Parameters
    ----------
    data_arr : np.array
        1D array of data
        
    Returns
    -------
    callable
        Quantile function
    """
    return lambda q, data_arr=data_arr: quantile(data_arr, q, method='inverted_cdf')


def uniform_ppf(left, right):
    """
    Create a uniform quantile function.
    
    Parameters
    ----------
    left : float
        Left boundary
    right : float
        Right boundary
        
    Returns
    -------
    callable
        Quantile function
    """
    return lambda q, left=left, scale=right - left: uniform.ppf(q, loc=left, scale=scale)


def normal_ppf(mean, std):
    """
    Create a normal quantile function.
    
    Parameters
    ----------
    mean : float
        Mean of the distribution
    std : float
        Standard deviation of the distribution
        
    Returns
    -------
    callable
        Quantile function
    """
    return lambda q, mean=mean, std=std: norm.ppf(q, loc=mean, scale=std)


def relative_error(est, true):
    """
    Calculate relative error between estimate and true value.
    
    Parameters
    ----------
    est : float
        Estimated value
    true : float
        True value
        
    Returns
    -------
    float
        Relative error
    """
    return np.abs((est - true) / true)


def fit_power(x, y):
    """
    Fit a power law y = a * (x ** b) to data.
    
    Parameters
    ----------
    x : np.array
        Independent variable
    y : np.array
        Dependent variable
        
    Returns
    -------
    tuple
        (a, b) coefficients
    """
    # y = a * (x ** b)
    # ln(y) = ln(a) + b * ln(x)
    ln_x = np.log(x).reshape(-1, 1)
    ln_y = np.log(y).reshape(-1, 1)

    X = np.concatenate([np.ones((len(x), 1)), 
                        ln_x], 
                       axis=1)
    
    ab = (np.linalg.pinv(X) @ ln_y).flatten()
    
    return np.exp(ab[0]), ab[1]


def power_law(x, a, b):
    """
    Power law function: y = a * (x ** b).
    
    Parameters
    ----------
    x : float or np.array
        Input values
    a : float
        Coefficient
    b : float
        Exponent
        
    Returns
    -------
    float or np.array
        Output values
    """
    return a * (x ** b)


def expected_unif_distance(a, b, c, d, n):
    """
    Calculate expected uniform distance between two uniform distributions.
    
    Parameters
    ----------
    a : float
        Left boundary of first distribution
    b : float
        Right boundary of first distribution
    c : float
        Left boundary of second distribution
    d : float
        Right boundary of second distribution
    n : int
        Number of samples
        
    Returns
    -------
    float
        Expected distance
    """
    diff1 = b - a
    diff2 = d - c

    return (((diff1 - diff2)**2) / 3) + ((a - c)**2) + ((a - c) * (diff1 - diff2)) + ((diff1 * diff2) / (3 * (n + 1)))

