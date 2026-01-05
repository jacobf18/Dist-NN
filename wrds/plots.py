"""
Plotting Module for Distributional Nearest Neighbors (Dist-NN)

This module provides functions for generating visualizations related to Dist-NN predictions,
including histogram comparisons, error boxplots, and quantile function plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

from .dist_nn import (
    empirical_quantile_function,
    expectation,
    var,
    relative_error
)


def convert_to_hist(emp_fn, bin_edges):
    """
    Convert an empirical quantile function to histogram format.
    
    Args:
        emp_fn (function): Empirical quantile function
        bin_edges (np.array): Array of bin edges
    
    Returns:
        tuple: (bin_centers, bin_heights) arrays
    """
    # Compute the quantile values for each bin edge
    q = np.linspace(0, 1, 100)
    quantiles = emp_fn(q)
    
    # Compute the probability mass for each bin 
    bin_heights = np.zeros(len(bin_edges) - 1)
    for i in range(len(bin_edges) - 1):
        index_in_quantiles = np.searchsorted(quantiles, bin_edges[i])
        next_index = np.searchsorted(quantiles, bin_edges[i + 1])
        if next_index == len(quantiles):
            next_index = len(quantiles) - 1
        if index_in_quantiles == len(quantiles):
            index_in_quantiles = len(quantiles) - 1
        bin_heights[i] = q[next_index] - q[index_in_quantiles]
        
    bin_heights /= np.diff(bin_edges)
    # Compute the bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, bin_heights


def plot_histogram_comparison(
    ticker: str,
    b_fn,
    actual_data: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot histogram comparison showing All Data, 2 Months Prior, and Dist-NN predictions.
    
    Args:
        ticker (str): Ticker symbol
        b_fn: Barycenter quantile function
        actual_data (pd.DataFrame): DataFrame with 'value' and 'ann_datetime' columns
        cutoff_date (pd.Timestamp): Cutoff date for filtering data
        save_path (str, optional): Path to save the figure. If None, figure is not saved.
        figsize (tuple): Figure size (width, height)
    """
    actual = actual_data['value'].values
    filtered_data = actual_data[actual_data['ann_datetime'] < cutoff_date]['value'].values
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.4)
    
    # Plot all data histogram
    sns.histplot(data=actual, label='All Data', stat='density', ax=ax)
    
    # Get the bin edges from the first histogram
    bin_edges = [p.get_x() + p.get_width() for p in ax.patches]
    bin_edges = [ax.patches[0].get_x()] + bin_edges
    
    # Plot filtered data histogram
    sns.histplot(filtered_data, bins=bin_edges, label='2 Months Prior', stat='density', ax=ax)
    
    # Plot Dist-NN prediction
    bin_centers, bin_heights = convert_to_hist(b_fn, np.array(bin_edges))
    ax.bar(bin_centers, bin_heights, width=np.diff(bin_edges), 
           alpha=0.6, align='center', edgecolor="black", linewidth=1.2, label='Dist-NN')
    
    plt.xlabel('EPS')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig, ax


def plot_error_boxplots(
    test_dict_distnn: Dict,
    cutoff_date: pd.Timestamp,
    alpha: float = 0.05,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 5),
    ylim_top: float = 150
):
    """
    Plot boxplots comparing Dist-NN errors vs baseline errors for different metrics.
    
    Args:
        test_dict_distnn (dict): Dictionary with ticker keys and prediction data
        cutoff_date (pd.Timestamp): Cutoff date for filtering data
        alpha (float): Significance level for VaR calculation (default: 0.05)
        save_path (str, optional): Path to save the figure. If None, figure is not saved.
        figsize (tuple): Figure size (width, height)
        ylim_top (float): Top limit for y-axis
    """
    mean_errors = []
    median_errors = []
    std_errors = []
    VaR_errors = []
    
    mean_errors_baseline = []
    median_errors_baseline = []
    std_errors_baseline = []
    VaR_errors_baseline = []
    
    for t, sub_dict in test_dict_distnn.items():
        b_fn = sub_dict['b_fn']
        actual_data = sub_dict['actual_data']
        
        cutoff_data = actual_data[actual_data['ann_datetime'] < cutoff_date]
        
        est_mean = expectation(b_fn)
        est_median = b_fn(np.array([0.5]))[0]
        est_std = np.sqrt(var(b_fn))
        est_VaR = np.quantile(-1 * b_fn(np.linspace(0, 1, 1000)), 1 - alpha)
        
        baseline_est_mean = cutoff_data['value'].mean()
        baseline_est_median = cutoff_data['value'].median()
        baseline_est_std = cutoff_data['value'].std()
        baseline_est_VaR = np.quantile(-1 * cutoff_data['value'].values, 1 - alpha)
        
        empirical_fn = empirical_quantile_function(np.sort(actual_data['value'].values))
        actual_mean = actual_data['value'].mean()
        actual_median = actual_data['value'].median()
        actual_std = np.sqrt(var(empirical_fn))
        actual_VaR = np.quantile(-1 * actual_data['value'].values, 1 - alpha)
        
        mean_errors.append(relative_error(est_mean, actual_mean) * 100)
        median_errors.append(relative_error(est_median, actual_median) * 100)
        std_errors.append(relative_error(est_std, actual_std) * 100)
        VaR_errors.append(relative_error(est_VaR, actual_VaR) * 100)
        
        mean_errors_baseline.append(relative_error(baseline_est_mean, actual_mean) * 100)
        median_errors_baseline.append(relative_error(baseline_est_median, actual_median) * 100)
        std_errors_baseline.append(relative_error(baseline_est_std, actual_std) * 100)
        VaR_errors_baseline.append(relative_error(baseline_est_VaR, actual_VaR) * 100)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=[
        mean_errors, mean_errors_baseline,
        median_errors, median_errors_baseline,
        std_errors, std_errors_baseline,
        VaR_errors, VaR_errors_baseline
    ], ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.4)
    ax.set_ylim(bottom=0, top=ylim_top)
    ax.set_xticklabels([
        "Mean\n(Dist-NN)", "Mean\n(Baseline)",
        "Median\n(Dist-NN)", "Median\n(Baseline)",
        "Std. Dev\n(Dist-NN)", "Std. Dev.\n(Baseline)",
        "VaR(5%)\n(Dist-NN)", "VaR(5%)\n(Baseline)"
    ])
    plt.ylabel("Relative Error (%)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig, ax


def plot_quantile_comparison(
    b_fn,
    actual_data: pd.DataFrame,
    n_quantiles: int = 100,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot quantile function comparison between barycenter and actual distribution.
    
    Args:
        b_fn: Barycenter quantile function
        actual_data (pd.DataFrame): DataFrame with 'value' column
        n_quantiles (int): Number of quantile points to plot (default: 100)
        save_path (str, optional): Path to save the figure. If None, figure is not saved.
        figsize (tuple): Figure size (width, height)
    """
    quantiles = np.linspace(0, 1, n_quantiles)
    
    # Get actual quantile function
    actual_sorted = np.sort(actual_data['value'].values)
    actual_qfn = empirical_quantile_function(actual_sorted)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(quantiles, b_fn(quantiles), label='Barycenter')
    ax.plot(quantiles, actual_qfn(quantiles), label='Actual')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.4)
    
    plt.xlabel('Quantile')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig, ax


def plot_individual_metric_boxplot(
    dist_errors: List[float],
    baseline_errors: List[float],
    metric_name: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    ylim_top: Optional[float] = None
):
    """
    Plot boxplot comparing Dist-NN vs baseline for a single metric.
    
    Args:
        dist_errors (list): List of Dist-NN relative errors (as percentages)
        baseline_errors (list): List of baseline relative errors (as percentages)
        metric_name (str): Name of the metric (e.g., "Mean", "Median")
        save_path (str, optional): Path to save the figure. If None, figure is not saved.
        figsize (tuple): Figure size (width, height)
        ylim_top (float, optional): Top limit for y-axis. If None, auto-scales.
    """
    errors_total = np.array([dist_errors, baseline_errors]).T
    labels = ["Dist-NN", "Baseline"]
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=errors_total, ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.4)
    
    if ylim_top is not None:
        ax.set_ylim(bottom=0, top=ylim_top)
    else:
        ax.set_ylim(bottom=0)
    
    ax.set_xticklabels(labels)
    plt.ylabel("Relative Error (%)")
    plt.title(f"{metric_name} Comparison")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig, ax

