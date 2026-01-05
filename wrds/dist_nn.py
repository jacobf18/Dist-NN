"""
Distributional Nearest Neighbors (Dist-NN) Module

This module implements distributional nearest neighbors for earnings forecast prediction.
It includes functions for:
- Computing Wasserstein distances between distributions
- Empirical quantile functions and barycenters
- Finding similar tickers/quarters based on distributional similarity
- Hyperparameter optimization
- Distributional prediction using barycenters
"""

import numpy as np
import pandas as pd
import ot
from collections import defaultdict
from hyperopt import hp, fmin, tpe
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def wasserstein2(sample1, sample2):
    """
    Compute the squared 2-Wasserstein distance between two samples.
    
    Args:
        sample1 (np.array): 2D array of samples from the first distribution (shape: [n_samples, n_dims])
        sample2 (np.array): 2D array of samples from the second distribution (shape: [m_samples, n_dims])
    
    Returns:
        float: Squared 2-Wasserstein distance
    """
    C = ot.dist(sample1, sample2, metric='sqeuclidean')
    # Uniform weights for each empirical distribution
    a = np.ones(len(sample1)) / len(sample1)
    b = np.ones(len(sample2)) / len(sample2)
    
    # Compute optimal transport plan and Wasserstein distance
    P = ot.emd(a, b, C)
    return np.sum(P * C)


def empirical_quantile_function(samples):
    """
    Create an empirical quantile function from sorted samples.
    
    Args:
        samples (np.array): Sorted 1D array of samples
        
    Returns:
        function: Quantile function that takes quantile values (0-1) and returns quantile values
    """
    samples_diff = np.concatenate([np.array(samples[0]).reshape(1), np.diff(samples)])
    
    def quantile_function(q):
        # Compute the empirical CDF values
        n = len(samples)
        cdf = np.arange(1, n + 1) / n
        # Use broadcasting to calculate the Heaviside contributions
        heaviside_matrix = np.heaviside(np.expand_dims(q, 1) - np.expand_dims(cdf, 0), 0.0)
        # Add a column of ones to the left of the Heaviside matrix
        first_col = np.ones(heaviside_matrix.shape[0]).reshape(-1, 1)
        heaviside_matrix = np.concatenate([first_col, heaviside_matrix], axis=1)
        # Remove the last column of Heaviside_matrix
        heaviside_matrix = heaviside_matrix[:, :-1]
        # Compute quantile values by summing contributions
        quantile_values = (heaviside_matrix @ samples_diff)
        return quantile_values
    
    return quantile_function


def linear_combination(quantile_fns, weights):
    """
    Create a linear combination of quantile functions.
    
    Args:
        quantile_fns (list): List of quantile functions
        weights (list): List of weights for each function
        
    Returns:
        function: Combined quantile function
    """
    def lin_comb_fn(quantiles):
        quantile_values = np.stack([fn(quantiles) for fn in quantile_fns])
        lin_comb_values = np.sum(np.expand_dims(weights, 1) * quantile_values, axis=0)
        return lin_comb_values
    return lin_comb_fn


def barycenter(quantile_fns):
    """
    Compute the barycenter (average) of quantile functions.
    
    Args:
        quantile_fns (list): List of quantile functions
        
    Returns:
        function: Barycenter quantile function
    """
    def lin_comb_fn(quantiles):
        quantile_values = np.stack([fn(quantiles) for fn in quantile_fns])
        lin_comb_values = np.sum(quantile_values, axis=0) / len(quantile_fns)
        return lin_comb_values
    return lin_comb_fn


def expectation(quantile_func, N=1000):
    """
    Compute the expectation (mean) of a distribution from its quantile function.
    
    Args:
        quantile_func (function): Quantile function
        N (int): Number of points for numerical integration (default: 1000)
        
    Returns:
        float: Expected value
    """
    x = np.linspace(0, 1, N)
    return np.trapz(quantile_func(x), x=x)


def var(quantile_func, N=1000):
    """
    Compute the variance of a distribution from its quantile function.
    
    Args:
        quantile_func (function): Quantile function
        N (int): Number of points for numerical integration (default: 1000)
        
    Returns:
        float: Variance
    """
    x = np.linspace(0, 1, N)
    mean = expectation(quantile_func, N)
    return np.trapz((quantile_func(x) - mean) ** 2, x=x)


def squared_diff(quantile_func1, quantile_func2, N=1000):
    """
    Compute the squared difference between two quantile functions.
    
    Args:
        quantile_func1 (function): First quantile function
        quantile_func2 (function): Second quantile function
        N (int): Number of points for numerical integration (default: 1000)
        
    Returns:
        float: Squared difference
    """
    x = np.linspace(0, 1, N)
    return np.trapz((quantile_func1(x) - quantile_func2(x)) ** 2, x=x)


def relative_error(est, true):
    """
    Compute relative error between estimate and true value.
    
    Args:
        est (float): Estimated value
        true (float): True value
        
    Returns:
        float: Relative error
    """
    return np.abs((est - true) / true)


def get_dist(ticker, year, quarter, cutoff_date, quarterly_actual, quarterly_data, 
             user_user=True, tickers=None):
    """
    Compute Wasserstein distances from a given cell to all other cells in the same column/row.
    
    Args:
        ticker (str): Ticker symbol
        year (int): Year
        quarter (int): Quarter (1-4)
        cutoff_date (pd.Timestamp): Cutoff date for filtering data (only use data before this)
        quarterly_actual (dict): Dictionary of actual values
        quarterly_data (dict): Dictionary of forecast data
        user_user (bool): If True, compute distances to other tickers (same quarter). 
                         If False, compute distances to other quarters (same ticker).
        tickers (list): Optional list of ticker symbols for user_user mode
    
    Returns:
        dict: Dictionary mapping (ticker, year, quarter) or (year, quarter) to Wasserstein distance
    """
    from download_data import get_rows_cols
    
    rows, _ = get_rows_cols(user_user, tickers=tickers)
    data = quarterly_data.get((ticker, year, quarter))
    
    if data is None:
        return {}
    
    # Filter by the announcement datetime
    filtered_data = data[data['ann_datetime'] < cutoff_date]
    sample1 = filtered_data['value'].values.reshape(-1, 1)
    
    if sample1.shape[0] == 0:
        return None
    
    dists = dict()
    actual = quarterly_actual.get((ticker, year, quarter))
    
    if actual is None:
        raise Exception(f"No actual data for {ticker} {year} {quarter}")
    
    if rows is None or len(rows) == 0:
        return {}
    
    for row in rows:
        if user_user:
            t = row
            y, q = year, quarter
        else:
            t = ticker
            y, q = row
        
        raw_data = quarterly_data.get((t, y, q))
        other_actual = quarterly_actual.get((t, y, q))
        
        if raw_data is None or other_actual is None:
            continue
        
        filtered_data = raw_data[raw_data['ann_datetime'] < cutoff_date]
        if filtered_data.shape[0] == 0:
            continue
        
        sample2 = filtered_data['value'].values.reshape(-1, 1)
        dists[row] = wasserstein2(sample1, sample2)
    
    return dists


def get_avg_dist_trains(train_cells, cutoff_date, average_cols, quarterly_actual, quarterly_data, user_user=True):
    """
    For each training cell, get the average distances from the average columns/rows.
    
    Args:
        train_cells (list): List of (ticker, year, quarter) tuples for training
        cutoff_date (pd.Timestamp): Cutoff date for filtering data
        average_cols (list): List of columns/rows to average over (tickers if user_user=True, quarters if False)
        quarterly_actual (dict): Dictionary of actual values
        quarterly_data (dict): Dictionary of forecast data
        user_user (bool): If True, average over tickers. If False, average over quarters.
    
    Returns:
        dict: Dictionary mapping train_cell to dict of (neighbor_cell, avg_distance) pairs
    """
    train_cells_avg_dists = defaultdict(dict)
    
    for cell in train_cells:
        t, y, q = cell
        row_union = set()
        dists = {}
        
        for col in average_cols:
            if user_user:
                dist = get_dist(col, y, q, cutoff_date, quarterly_actual, quarterly_data, user_user=False)
            else:
                dist = get_dist(t, col[0], col[1], cutoff_date, quarterly_actual, quarterly_data, user_user=True)
            
            if dist is None:
                continue
            
            row_union = row_union.union(dist.keys())
            dists[col] = dist
        
        for row in row_union:
            if user_user:
                other_cell = (row, y, q)
            else:
                other_cell = (t, row[0], row[1])
            
            avg_dist = np.mean([dist[row] for dist in dists.values() if row in dist])
            train_cells_avg_dists[cell][other_cell] = avg_dist
    
    return train_cells_avg_dists


def optimize_eta(average_cols, train_cells, cutoff_date, quarterly_data, quarterly_actual, user_user=True, verbose=False):
    """
    Optimize the threshold parameter eta for selecting neighbors.
    
    Args:
        average_cols (list): List of columns/rows to average over
        train_cells (list): List of training cells (ticker, year, quarter)
        cutoff_date (pd.Timestamp): Cutoff date for filtering data
        quarterly_data (dict): Dictionary of forecast data
        quarterly_actual (dict): Dictionary of actual values
        user_user (bool): If True, average over tickers. If False, average over quarters.
        verbose (bool): Whether to print optimization progress
    
    Returns:
        float: Optimal eta threshold value
    """
    train_cells_avg_dists = get_avg_dist_trains(
        train_cells, cutoff_date, average_cols, quarterly_actual, quarterly_data, user_user
    )
    
    train_cells_empirical_quantile_fns = {
        (t, y, q): empirical_quantile_function(np.sort(quarterly_data[t, y, q]['value'].values))
        for t, y, q in train_cells
    }
    
    def obj(params):
        eta = params['eta']
        total_error = 0
        
        for cell in train_cells:
            avg_dist_dict = train_cells_avg_dists[cell]
            neighbors = [c for c, dist in avg_dist_dict.items() if dist <= eta and c != cell]
            
            sample_fns = []
            for c in neighbors:
                sample = quarterly_data[c[0], c[1], c[2]]
                sample_fns.append(empirical_quantile_function(np.sort(sample['value'].values)))
            
            if len(sample_fns) == 0:
                total_error += 1000
                continue
            
            b_fn = barycenter(sample_fns)
            total_error += squared_diff(b_fn, train_cells_empirical_quantile_fns[cell])
        
        return total_error / len(train_cells)
    
    # Optimize using hyperopt
    best_eta = fmin(
        fn=obj,
        verbose=verbose,
        space={'eta': hp.loguniform('eta', -10, 2)},
        algo=tpe.suggest,
        max_evals=50
    )
    
    return best_eta['eta']


def get_similar_tickers(ticker, year, quarter, cutoff_date, quarterly_actual, quarterly_data, 
                       user_user=False, verbose=False, tickers=None):
    """
    Find similar tickers based on distributional distances across quarters.
    
    Args:
        ticker (str): Ticker symbol to find similar tickers for
        year (int): Year
        quarter (int): Quarter (1-4)
        cutoff_date (pd.Timestamp): Cutoff date for filtering data
        quarterly_actual (dict): Dictionary of actual values
        quarterly_data (dict): Dictionary of forecast data
        user_user (bool): Whether to use user-user mode (default: False for item-item)
        verbose (bool): Whether to show progress bar
        tickers (list): Optional list of ticker symbols
    
    Returns:
        dict: Dictionary mapping ticker to distance vector
    """
    from download_data import get_rows_cols
    
    rows, cols = get_rows_cols(user_user, tickers=tickers)
    current_row = (year, quarter)
    
    if user_user:
        row_index = rows.index(ticker) if ticker in rows else 0
    else:
        row_index = rows.index(current_row) if current_row in rows else 0
    
    actual = quarterly_actual.get((ticker, year, quarter))
    if actual is None:
        raise Exception(f"No actual data for {ticker, year, quarter}")
    
    distance_vectors = dict()
    iterator = tqdm(cols) if verbose else cols
    
    for t in iterator:
        distance = np.zeros(row_index)
        try:
            dist = get_dist(t, year, quarter, cutoff_date, quarterly_actual, quarterly_data, user_user=False)
            if dist is None:
                continue
            
            for i in range(0, row_index):
                if rows[i] in dist:
                    distance[i] = dist[rows[i]]
                else:
                    distance[i] = np.nan
            
            distance_vectors[t] = distance
        except Exception:
            continue
    
    return distance_vectors


def predict_distribution(ticker, year, quarter, cutoff_date, quarterly_data, quarterly_actual,
                        similar_tickers, dates_average, eta, user_user=False):
    """
    Predict the distribution for a given cell using distributional nearest neighbors.
    
    Args:
        ticker (str): Ticker symbol
        year (int): Year
        quarter (int): Quarter (1-4)
        cutoff_date (pd.Timestamp): Cutoff date for filtering data
        quarterly_data (dict): Dictionary of forecast data
        quarterly_actual (dict): Dictionary of actual values
        similar_tickers (list): List of similar tickers to use
        dates_average (list): List of (year, quarter) tuples to average over
        eta (float): Distance threshold for selecting neighbors
        user_user (bool): Whether to use user-user mode
    
    Returns:
        function: Predicted quantile function (barycenter)
    """
    # Get average distances
    train_cells_avg_dists = get_avg_dist_trains(
        [(ticker, year, quarter)], cutoff_date, similar_tickers, 
        quarterly_actual, quarterly_data, user_user
    )
    
    avg_dist_dict = train_cells_avg_dists.get((ticker, year, quarter), {})
    
    # Select neighbors based on threshold
    neighbors = [c for c, dist in avg_dist_dict.items() if dist <= eta]
    
    # Compute barycenter of neighbor distributions
    sample_fns = []
    for c in neighbors:
        sample = quarterly_data.get(c)
        if sample is None:
            continue
        filtered_data = sample[sample['ann_datetime'] < cutoff_date]
        if filtered_data.shape[0] == 0:
            continue
        sample_fns.append(empirical_quantile_function(np.sort(filtered_data['value'].values)))
    
    if len(sample_fns) == 0:
        return None
    
    return barycenter(sample_fns)


def test_train_wasserstein(test_cell, cutoff_date, quarterly_data, quarterly_actual, 
                           user_user=False, verbose=False, num_missing=10):
    """
    Test the distributional nearest neighbors method on missing data scenarios.
    
    Args:
        test_cell (tuple): (ticker, year, quarter) for the test cell
        cutoff_date (pd.Timestamp): Cutoff date for filtering data
        quarterly_data (dict): Dictionary of forecast data
        quarterly_actual (dict): Dictionary of actual values
        user_user (bool): Whether to use user-user mode
        verbose (bool): Whether to show progress
        num_missing (int): Number of missing tickers to test on
    
    Returns:
        dict: Dictionary mapping ticker to {'b_fn': quantile_function, 'actual_data': DataFrame}
    """
    ticker, year, quarter = test_cell
    
    from download_data import get_rows_cols
    # Get tickers from quarterly_data keys
    all_tickers = list(set([k[0] for k in quarterly_data.keys() if k[0] is not None]))
    _, cols = get_rows_cols(user_user, tickers=all_tickers)
    
    actual = quarterly_actual.get(test_cell)
    if actual is None:
        return None
    
    # Previous 8 quarters for averaging
    dates_average = []
    current_date = (year, quarter)
    for _ in range(8):
        dates_average.append(current_date)
        if current_date[1] == 1:
            current_date = (current_date[0] - 1, 4)
        else:
            current_date = (current_date[0], current_date[1] - 1)
    dates_average = dates_average[1:]  # exclude current quarter
    
    # Find missing tickers (those with mostly future data)
    missing_tickers = []
    for t in cols:
        other_actual = quarterly_actual.get((t, year, quarter))
        if other_actual is None:
            continue
        
        other_data = quarterly_data.get((t, year, quarter))
        if other_data is None:
            continue
        
        filtered_data = other_data[other_data['ann_datetime'] < cutoff_date]
        if filtered_data.shape[0] / other_data.shape[0] < 0.2 and filtered_data.shape[0] >= 1:
            missing_tickers.append(t)
    
    missing_tickers_subset = missing_tickers[:num_missing]
    missing_tickers = set(missing_tickers)
    
    output = dict()
    prev_date = dates_average[0]
    
    for missing_ticker in (pbar := tqdm(missing_tickers_subset, disable=not verbose)):
        if verbose:
            pbar.set_description(f"Processing {missing_ticker}")
        
        try:
            # Get similar tickers
            distance_vectors = get_similar_tickers(
                missing_ticker, prev_date[0], prev_date[1], cutoff_date,
                quarterly_actual, quarterly_data, user_user=user_user, verbose=False
            )
            
            if missing_ticker not in distance_vectors:
                continue
            
            cur_vec = distance_vectors[missing_ticker]
            similarity = []
            
            for t, vec in distance_vectors.items():
                if t in missing_tickers or t == missing_ticker:
                    continue
                
                other_data = quarterly_data.get((t, year, quarter))
                if other_data is None:
                    continue
                
                filtered_data = other_data[other_data['ann_datetime'] < cutoff_date]
                if filtered_data.shape[0] / other_data.shape[0] < 0.4 or filtered_data.shape[0] < 5:
                    continue
                
                similarity.append((t, np.nanmean((vec - cur_vec) ** 2)))
            
            similarity.sort(key=lambda x: x[1])
            similar_tickers = [t for t, _ in similarity[:30]]
            
            # Train cells (previous 4 quarters)
            train_cells = [(missing_ticker, y, q) for y, q in dates_average[:4]]
            
            # Optimize eta
            eta = optimize_eta(
                similar_tickers, train_cells, cutoff_date, 
                quarterly_data, quarterly_actual, user_user=user_user, verbose=False
            )
            
            # Calculate estimate
            train_cells_avg_dists = get_avg_dist_trains(
                [(missing_ticker, year, quarter)], cutoff_date, similar_tickers,
                quarterly_actual, quarterly_data, user_user
            )
            
            avg_dist_dict = train_cells_avg_dists.get((missing_ticker, year, quarter), {})
            neighbors = [c for c, dist in avg_dist_dict.items() if dist <= eta]
            
            sample_fns = []
            for c in neighbors:
                sample = quarterly_data.get(c)
                if sample is None:
                    continue
                filtered_data = sample[sample['ann_datetime'] < cutoff_date]
                if filtered_data.shape[0] == 0:
                    continue
                sample_fns.append(empirical_quantile_function(np.sort(filtered_data['value'].values)))
            
            if len(sample_fns) == 0:
                continue
            
            b_fn = barycenter(sample_fns)
            
            output[missing_ticker] = {
                'b_fn': b_fn,
                'actual_data': quarterly_data.get((missing_ticker, year, quarter))
            }
        except Exception as e:
            if verbose:
                print(f"Error for {missing_ticker}: {e}")
            continue
    
    return output

