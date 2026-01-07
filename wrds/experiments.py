"""
Experiments Module for Distributional Nearest Neighbors

This module provides functions for running various experiments from the notebook,
including doubly-robust estimator experiments and evaluation runs.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Optional
import random

from .dist_nn import (
    train_test_cell,
    get_earnings_time
)
from .download_data import get_rows_cols


def run_doubly_robust_experiment(
    tickers: List[str],
    dates: List[Tuple[int, int]],
    quarterly_data: dict,
    quarterly_actual: dict,
    output_file: str = 'doubly_robust_errors.csv',
    dates_per_ticker: int = 3,
    verbose: bool = False
):
    """
    Run doubly-robust estimator experiment on multiple tickers and dates.
    
    Args:
        tickers (list): List of ticker symbols to test
        dates (list): List of (year, quarter) tuples to sample from
        quarterly_data (dict): Dictionary of forecast data
        quarterly_actual (dict): Dictionary of actual values
        output_file (str): Path to output CSV file
        dates_per_ticker (int): Number of dates to sample per ticker (default: 3)
        verbose (bool): Whether to show progress
        
    Returns:
        pd.DataFrame: Results DataFrame
    """
    results = []
    file_exists = False
    
    for ticker in tqdm(tickers, desc="Doubly-robust experiment"):
        rand_dates = random.sample(dates, min(dates_per_ticker, len(dates)))
        for year, quarter in rand_dates:
            test_cell = (ticker, year, quarter)
            try:
                test_error, baseline_error = train_test_cell(
                    test_cell, quarterly_data, quarterly_actual, verbose=verbose
                )
                
                df = pd.DataFrame({
                    'ticker': [ticker],
                    'year': [year],
                    'quarter': [quarter],
                    'test_error': [test_error],
                    'base_error': [baseline_error]
                })
                
                if not file_exists:
                    df.to_csv(output_file, mode='w', index=False)
                    file_exists = True
                else:
                    df.to_csv(output_file, mode='a', index=False, header=False)
                
                results.append({
                    'ticker': ticker,
                    'year': year,
                    'quarter': quarter,
                    'test_error': test_error,
                    'base_error': baseline_error
                })
            except Exception as e:
                if verbose:
                    print(f"Error for {test_cell}: {e}")
                continue
    
    return pd.DataFrame(results)


def run_item_item_evaluation(
    tickers: List[str],
    year: int,
    quarter: int,
    quarterly_data: dict,
    quarterly_actual: dict,
    quarterly_means: dict,
    output_file: str = 'errors_item_item_small.csv',
    cutoff_date: Optional[pd.Timestamp] = None,
    verbose: bool = False
):
    """
    Run item-item evaluation on a set of tickers for a specific quarter.
    
    Args:
        tickers (list): List of ticker symbols to evaluate
        year (int): Year to evaluate
        quarter (int): Quarter to evaluate
        quarterly_data (dict): Dictionary of forecast data
        quarterly_actual (dict): Dictionary of actual values
        quarterly_means (dict): Dictionary of mean forecasts
        output_file (str): Path to output CSV file
        cutoff_date (pd.Timestamp, optional): Cutoff date for filtering data
        verbose (bool): Whether to show progress
        
    Returns:
        pd.DataFrame: Results DataFrame
    """
    from .dist_nn import optimize_eta, evaluate_eta_test
    
    results = []
    file_exists = False
    
    subset_tickers = set(tickers)
    
    for ticker in tqdm(subset_tickers, desc="Item-item evaluation"):
        train_cols = list(subset_tickers - {ticker})
        test_cols = [ticker]
        
        try:
            test_cell = (ticker, year, quarter)
            if quarterly_actual.get(test_cell) is None:
                continue
            
            eta = optimize_eta(
                train_cols, [test_cell], cutoff_date, quarterly_means, quarterly_actual,
                user_user=False, verbose=verbose
            )
            
            test_error = evaluate_eta_test(
                eta, ticker, year, quarter, train_cols, test_cols,
                quarterly_actual, quarterly_means, quarterly_means,
                cutoff_date=cutoff_date, user_user=False
            )
            
            base_error = evaluate_eta_test(
                0.0, ticker, year, quarter, train_cols, test_cols,
                quarterly_actual, quarterly_means, quarterly_means,
                cutoff_date=cutoff_date, user_user=False
            )
            
            df = pd.DataFrame({
                'ticker': [ticker],
                'eta': [eta],
                'test_error': [test_error],
                'base_error': [base_error]
            })
            
            if not file_exists:
                df.to_csv(output_file, mode='w', index=False)
                file_exists = True
            else:
                df.to_csv(output_file, mode='a', index=False, header=False)
            
            results.append({
                'ticker': ticker,
                'eta': eta,
                'test_error': test_error,
                'base_error': base_error
            })
        except Exception as e:
            if verbose:
                print(f"Error for {ticker}: {e}")
            continue
    
    return pd.DataFrame(results)


def run_user_user_evaluation(
    tickers: List[str],
    train_cols: List[Tuple[int, int]],
    test_cols: List[Tuple[int, int]],
    quarterly_data: dict,
    quarterly_actual: dict,
    quarterly_means: dict,
    output_file: str = 'test_errors_4_periods.csv',
    cutoff_date: Optional[pd.Timestamp] = None,
    verbose: bool = False
):
    """
    Run user-user evaluation on a set of tickers.
    
    Args:
        tickers (list): List of ticker symbols to evaluate
        train_cols (list): List of (year, quarter) tuples for training
        test_cols (list): List of (year, quarter) tuples for testing
        quarterly_data (dict): Dictionary of forecast data
        quarterly_actual (dict): Dictionary of actual values
        quarterly_means (dict): Dictionary of mean forecasts
        output_file (str): Path to output CSV file
        cutoff_date (pd.Timestamp, optional): Cutoff date for filtering data
        verbose (bool): Whether to show progress
        
    Returns:
        pd.DataFrame: Results DataFrame
    """
    from .dist_nn import optimize_eta, evaluate_eta_test
    
    results = []
    file_exists = False
    
    for ticker in tqdm(tickers, desc="User-user evaluation"):
        try:
            # Use first test column for optimization
            test_cell = (ticker, test_cols[0][0], test_cols[0][1])
            if quarterly_actual.get(test_cell) is None:
                continue
            
            eta = optimize_eta(
                train_cols, [test_cell], cutoff_date, quarterly_data, quarterly_actual,
                user_user=True, verbose=verbose
            )
            
            test_error = evaluate_eta_test(
                eta, ticker, test_cols[0][0], test_cols[0][1], train_cols, test_cols,
                quarterly_actual, quarterly_data, quarterly_means,
                cutoff_date=cutoff_date, user_user=True
            )
            
            base_error = evaluate_eta_test(
                0.0, ticker, test_cols[0][0], test_cols[0][1], train_cols, test_cols,
                quarterly_actual, quarterly_data, quarterly_means,
                cutoff_date=cutoff_date, user_user=True
            )
            
            df = pd.DataFrame({
                'ticker': [ticker],
                'eta': [eta],
                'test_error': [test_error],
                'base_error': [base_error]
            })
            
            if not file_exists:
                df.to_csv(output_file, mode='w', index=False)
                file_exists = True
            else:
                df.to_csv(output_file, mode='a', index=False, header=False)
            
            results.append({
                'ticker': ticker,
                'eta': eta,
                'test_error': test_error,
                'base_error': base_error
            })
        except Exception as e:
            if verbose:
                print(f"Error for {ticker}: {e}")
            continue
    
    return pd.DataFrame(results)


def calculate_avg_estimators(
    tickers: List[str],
    quarterly_data: dict
):
    """
    Calculate average number of estimators per ticker across all quarters.
    
    Args:
        tickers (list): List of ticker symbols
        quarterly_data (dict): Dictionary of forecast data
        
    Returns:
        dict: Dictionary mapping ticker to average number of estimators
    """
    _, cols = get_rows_cols(user_user=True)
    
    avg_estimator = {}
    for ticker in tickers:
        num_estimators = 0
        num_added = 0
        for col in cols:
            year, quarter = col
            data = quarterly_data.get((ticker, year, quarter))
            if data is None:
                continue
            num_estimators += data.shape[0]
            num_added += 1
        avg_estimator[ticker] = num_estimators / num_added if num_added > 0 else 0
    
    return avg_estimator

