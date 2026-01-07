"""
Evaluation Module for Distributional Nearest Neighbors

This module provides functions for evaluating Dist-NN using different windowing strategies:
- Rolling window
- Growing window
- Seasonal window
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Optional
import random

from .dist_nn import (
    optimize_eta,
    evaluate_eta_test
)
from .download_data import get_rows_cols


def rolling_window_evaluation(
    tickers: List[str],
    quarterly_data: dict,
    quarterly_actual: dict,
    quarterly_means: dict,
    window_size: int = 10,
    start_idx: int = 10,
    output_file: str = 'rolling_window_10.csv',
    cutoff_date: Optional[pd.Timestamp] = None,
    user_user: bool = True
):
    """
    Evaluate Dist-NN using a rolling window approach.
    
    Args:
        tickers (list): List of ticker symbols to evaluate
        quarterly_data (dict): Dictionary of forecast data
        quarterly_actual (dict): Dictionary of actual values
        quarterly_means (dict): Dictionary of mean forecasts
        window_size (int): Size of rolling window (default: 10)
        start_idx (int): Starting index for evaluation (default: 10)
        output_file (str): Path to output CSV file
        cutoff_date (pd.Timestamp, optional): Cutoff date for filtering data
        user_user (bool): If True, use user-user mode. If False, use item-item mode.
        
    Returns:
        pd.DataFrame: Results DataFrame
    """
    _, cols = get_rows_cols(user_user)
    
    results = []
    file_exists = False
    
    for j in range(start_idx, len(cols)):
        train_cols = cols[j - window_size:j]
        test_col = cols[j]
        year, quarter = test_col
        
        for ticker in tqdm(tickers, desc=f"Window {j-start_idx+1}/{len(cols)-start_idx}"):
            try:
                if user_user:
                    test_cell = (ticker, year, quarter)
                else:
                    test_cell = (ticker, year, quarter)
                
                if quarterly_actual.get(test_cell) is None:
                    continue
                
                eta = optimize_eta(
                    train_cols, [test_cell], cutoff_date, quarterly_data, quarterly_actual,
                    user_user=user_user, verbose=False
                )
                
                test_error = evaluate_eta_test(
                    eta, ticker, year, quarter, train_cols, [test_col],
                    quarterly_actual, quarterly_data, quarterly_means,
                    cutoff_date=cutoff_date, user_user=user_user
                )
                
                base_error = evaluate_eta_test(
                    0.0, ticker, year, quarter, train_cols, [test_col],
                    quarterly_actual, quarterly_data, quarterly_means,
                    cutoff_date=cutoff_date, user_user=user_user
                )
                
                df = pd.DataFrame({
                    'ticker': [ticker],
                    'year': [year],
                    'quarter': [quarter],
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
                    'year': year,
                    'quarter': quarter,
                    'eta': eta,
                    'test_error': test_error,
                    'base_error': base_error
                })
            except Exception as e:
                if len(results) == 0:  # Only print first few errors
                    print(f"Error for {ticker} {year} Q{quarter}: {e}")
                continue
    
    return pd.DataFrame(results)


def growing_window_evaluation(
    tickers: List[str],
    quarterly_data: dict,
    quarterly_actual: dict,
    quarterly_means: dict,
    start_idx: int = 10,
    output_file: str = 'growing_window.csv',
    cutoff_date: Optional[pd.Timestamp] = None,
    user_user: bool = True
):
    """
    Evaluate Dist-NN using a growing window approach.
    
    Args:
        tickers (list): List of ticker symbols to evaluate
        quarterly_data (dict): Dictionary of forecast data
        quarterly_actual (dict): Dictionary of actual values
        quarterly_means (dict): Dictionary of mean forecasts
        start_idx (int): Starting index for evaluation (default: 10)
        output_file (str): Path to output CSV file
        cutoff_date (pd.Timestamp, optional): Cutoff date for filtering data
        user_user (bool): If True, use user-user mode. If False, use item-item mode.
        
    Returns:
        pd.DataFrame: Results DataFrame
    """
    _, cols = get_rows_cols(user_user)
    
    results = []
    file_exists = False
    
    for j in range(start_idx, len(cols)):
        train_cols = cols[:j]  # All previous quarters
        test_col = cols[j]
        year, quarter = test_col
        
        for ticker in tqdm(tickers, desc=f"Growing window {j-start_idx+1}/{len(cols)-start_idx}"):
            try:
                if user_user:
                    test_cell = (ticker, year, quarter)
                else:
                    test_cell = (ticker, year, quarter)
                
                if quarterly_actual.get(test_cell) is None:
                    continue
                
                eta = optimize_eta(
                    train_cols, [test_cell], cutoff_date, quarterly_data, quarterly_actual,
                    user_user=user_user, verbose=False
                )
                
                test_error = evaluate_eta_test(
                    eta, ticker, year, quarter, train_cols, [test_col],
                    quarterly_actual, quarterly_data, quarterly_means,
                    cutoff_date=cutoff_date, user_user=user_user
                )
                
                base_error = evaluate_eta_test(
                    0.0, ticker, year, quarter, train_cols, [test_col],
                    quarterly_actual, quarterly_data, quarterly_means,
                    cutoff_date=cutoff_date, user_user=user_user
                )
                
                df = pd.DataFrame({
                    'ticker': [ticker],
                    'year': [year],
                    'quarter': [quarter],
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
                    'year': year,
                    'quarter': quarter,
                    'eta': eta,
                    'test_error': test_error,
                    'base_error': base_error
                })
            except Exception as e:
                if len(results) == 0:  # Only print first few errors
                    print(f"Error for {ticker} {year} Q{quarter}: {e}")
                continue
    
    return pd.DataFrame(results)


def seasonal_window_evaluation(
    tickers: List[str],
    quarterly_data: dict,
    quarterly_actual: dict,
    quarterly_means: dict,
    train_years: List[int],
    test_year: int,
    output_file: str = 'seasonal_test.csv',
    cutoff_date: Optional[pd.Timestamp] = None,
    user_user: bool = True
):
    """
    Evaluate Dist-NN using a seasonal window approach (same quarter, different years).
    
    Args:
        tickers (list): List of ticker symbols to evaluate
        quarterly_data (dict): Dictionary of forecast data
        quarterly_actual (dict): Dictionary of actual values
        quarterly_means (dict): Dictionary of mean forecasts
        train_years (list): List of years to use for training
        test_year (int): Year to test on
        output_file (str): Path to output CSV file
        cutoff_date (pd.Timestamp, optional): Cutoff date for filtering data
        user_user (bool): If True, use user-user mode. If False, use item-item mode.
        
    Returns:
        pd.DataFrame: Results DataFrame
    """
    results = []
    file_exists = False
    
    for quarter in range(1, 5):
        train_cols = [(y, quarter) for y in train_years]
        test_col = (test_year, quarter)
        
        for ticker in tqdm(tickers, desc=f"Seasonal Q{quarter}"):
            try:
                if user_user:
                    test_cell = (ticker, test_year, quarter)
                else:
                    test_cell = (ticker, test_year, quarter)
                
                if quarterly_actual.get(test_cell) is None:
                    continue
                
                eta = optimize_eta(
                    train_cols, [test_cell], cutoff_date, quarterly_data, quarterly_actual,
                    user_user=user_user, verbose=False
                )
                
                test_error = evaluate_eta_test(
                    eta, ticker, test_year, quarter, train_cols, [test_col],
                    quarterly_actual, quarterly_data, quarterly_means,
                    cutoff_date=cutoff_date, user_user=user_user
                )
                
                base_error = evaluate_eta_test(
                    0.0, ticker, test_year, quarter, train_cols, [test_col],
                    quarterly_actual, quarterly_data, quarterly_means,
                    cutoff_date=cutoff_date, user_user=user_user
                )
                
                df = pd.DataFrame({
                    'ticker': [ticker],
                    'year': [test_year],
                    'quarter': [quarter],
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
                    'year': test_year,
                    'quarter': quarter,
                    'eta': eta,
                    'test_error': test_error,
                    'base_error': base_error
                })
            except Exception as e:
                if len(results) == 0:  # Only print first few errors
                    print(f"Error for {ticker} {test_year} Q{quarter}: {e}")
                continue
    
    return pd.DataFrame(results)

