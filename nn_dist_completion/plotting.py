"""
Plotting functions for visualization and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.stats
import pandas as pd
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple

from .utils import wasserstein2, empirical_quantile, fit_power, power_law
from .config import LINE_STYLES, MARKERS


def plot_setup_visualization(save_path="figures/setup.pdf"):
    """
    Create visualization of the experimental setup.
    
    Parameters
    ----------
    save_path : str, optional
        Path to save the figure (default: "figures/setup.pdf")
    """
    fig, axes = plt.subplots(3, 6, figsize=(15, 7.5), sharey='row', sharex='col', layout="compressed")
    min_xs = {i: np.inf for i in range(6)}
    max_xs = {i: -np.inf for i in range(6)}
    skip = set([(0, 0), (2, 4), (1, 5), (2, 2)])
    
    for i in range(3):
        for j in range(6):
            if (i, j) in skip:
                continue
            data = np.random.normal(i + 10, (j + 1) * 0.5, 100)
            sns.histplot(data, ax=axes[i, j], stat='density', zorder=1)
            min_xs[j] = min(min_xs[j], np.min(data))
            max_xs[j] = max(max_xs[j], np.max(data))

    for i in range(3):
        for j in range(6):
            x = np.linspace(min_xs[j], max_xs[j], 100)
            pdf = scipy.stats.norm.pdf(x, i + 10, (j + 1) * 0.5)
            axes[i, j].plot(x, pdf,
                           linewidth=3,
                           alpha=0.5,
                           color='black',
                           zorder=0)
            axes[i, j].spines['top'].set_visible(False)
            axes[i, j].spines['right'].set_visible(False)

    axes[0, 0].set_title("English\n(No digital resources)")
    axes[0, 1].set_title("English\n(Digital resources)")
    axes[0, 2].set_title("Mathematics\n(No digital resources)")
    axes[0, 3].set_title("Mathematics\n(Digital resources)")
    axes[0, 4].set_title("Science\n(No digital resources)")
    axes[0, 5].set_title("Science\n(Digital resources)")
    axes[0, 0].set_ylabel("School A")
    axes[1, 0].set_ylabel("School B")
    axes[2, 0].set_ylabel("School C")
    plt.savefig(save_path)
    plt.close()


def plot_cdf_comparison(est_dist, true_dist_data, true_dist_ppf, n_samples=2, 
                       dist_type='normal', save_path=None):
    """
    Plot CDF comparison between estimated and true distributions.
    
    Parameters
    ----------
    est_dist : np.ndarray
        Estimated distribution samples
    true_dist_data : np.ndarray
        True distribution samples
    true_dist_ppf : callable
        True distribution quantile function
    n_samples : int, optional
        Number of random samples to plot (default: 2)
    dist_type : str, optional
        Distribution type ('normal' or 'uniform') (default: 'normal')
    save_path : str, optional
        Path to save the figure
    """
    x_min = min(np.min(true_dist_data), np.min(est_dist))
    x_max = max(np.max(true_dist_data), np.max(est_dist))
    x = np.linspace(x_min, x_max, num=1000)
    
    if dist_type == 'normal':
        # Extract parameters from true_dist_ppf if possible
        # For now, assume we have mean and std from context
        y = scipy.stats.norm.cdf(x, np.mean(true_dist_data), np.std(true_dist_data))
    else:
        y = scipy.stats.uniform.cdf(x, np.min(true_dist_data), 
                                    np.max(true_dist_data) - np.min(true_dist_data))
    
    mpl.rcParams['legend.fontsize'] = 20
    
    plt.ecdf(est_dist, linewidth=3, alpha=0.9)
    for _ in range(n_samples):
        if dist_type == 'normal':
            sample = np.random.normal(np.mean(true_dist_data), np.std(true_dist_data), len(est_dist))
        else:
            sample = np.random.uniform(np.min(true_dist_data), np.max(true_dist_data), len(est_dist))
        plt.ecdf(sample, linewidth=2, alpha=0.8)
    
    plt.plot(x, y, '--', c='k', linewidth=1.5)
    plt.legend(["Synth. Sample"] + [f"Rand. Sample {i + 1}" for i in range(n_samples)] + ["True Dist."], 
               loc='lower right')
    plt.xlabel("$x$")
    plt.ylabel("CDF($x$)")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.4)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_error_vs_samples(errors_df, rows, error_type='Estimation Error', 
                          save_path="figures/error-samples-normal-N_2.pdf"):
    """
    Plot error vs number of samples.
    
    Parameters
    ----------
    errors_df : pd.DataFrame
        DataFrame with error data
    rows : list
        List of row counts to plot
    error_type : str, optional
        Column name for error (default: 'Estimation Error')
    save_path : str, optional
        Path to save the figure
    """
    label_size = 20 / 0.8
    mpl.rcParams['lines.markersize'] = 7
    mpl.rcParams['grid.linewidth'] = 2.5
    mpl.rcParams['legend.fontsize'] = label_size
    mpl.rcParams['xtick.labelsize'] = label_size 
    mpl.rcParams['ytick.labelsize'] = label_size 
    mpl.rcParams['axes.labelsize'] = label_size
    mpl.rcParams['axes.titlesize'] = label_size
    mpl.rcParams['figure.titlesize'] = label_size

    plt.figure()
    ax = plt.gca()

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.4)

    ls = []
    b = []
    
    # Calculate baseline error (random sample)
    baseline_error = errors_df[error_type].mean()  # approximate baseline
    
    l = ax.hlines(baseline_error, 0, max(errors_df['Samples']), 
                  colors='r', linestyles='dotted', linewidth=4.0)
    ls.append(l)

    for i, r in enumerate(rows):
        df = errors_df[errors_df['Rows'] == r]
        # filter out the rows with less than 2 neighbors
        df = df[df['Neighbors'] >= 2]
        mean = df.groupby('Samples').mean()

        l1 = ax.scatter(mean.index, mean[error_type], marker=MARKERS[i])
        c, d = fit_power(np.array(mean.index), np.array(mean[error_type]))
        l2, = ax.plot(mean.index, power_law(mean.index, c, d), 
                      LINE_STYLES[i], linewidth=4.0, alpha=0.5)
        b.append(d)

        ls.append((l1, l2))

    ax.set_ylim(bottom=1e-6)
    ax.set_xlabel("$n$")
    ax.set_ylabel("$W_2^2$ Error")
    labels = ["Random Sample"] + [f"N={m}, " + r"n$^{%.2f}$" % (d) for m, d in zip(rows, b)]
    ax.legend(handles=ls, labels=labels, loc='lower left', handletextpad=0.0,
              handler_map={tuple: HandlerTuple(ndivide=None)}, borderpad=0.1, labelspacing=0)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_error_vs_rows(errors_dict, rows, cols, samples, save_path="figures/error_rows.pdf"):
    """
    Plot error vs number of rows.
    
    Parameters
    ----------
    errors_dict : dict
        Dictionary mapping (M, N, n) to error tuples
    rows : list
        List of row counts
    cols : int
        Number of columns
    samples : int
        Number of samples
    save_path : str, optional
        Path to save the figure
    """
    errors_avg = {k: np.mean(v[0]) for k, v in errors_dict.items()}
    errors_std = {k: 2 * np.std(v[0]) for k, v in errors_dict.items()}

    y1 = np.array([errors_avg[m, cols, samples] for m in rows])
    y_err = np.array([errors_std[m, cols, samples] for m in rows])

    plt.figure()

    label_size = 20 / 0.8
    mpl.rcParams['lines.markersize'] = 7
    mpl.rcParams['grid.linewidth'] = 2.5
    mpl.rcParams['legend.fontsize'] = label_size
    mpl.rcParams['xtick.labelsize'] = label_size 
    mpl.rcParams['ytick.labelsize'] = label_size 
    mpl.rcParams['axes.labelsize'] = label_size
    mpl.rcParams['axes.titlesize'] = label_size
    mpl.rcParams['figure.titlesize'] = label_size

    # Calculate baseline
    baseline = np.mean([errors_avg.get((m, cols, samples), 0) for m in rows])
    plt.hlines(baseline, 0, max(rows), colors='r', linestyles='dotted', linewidth=4.0)
    plt.errorbar(rows, y1, yerr=y_err, fmt='o', color='b', linewidth=1.0)

    a, b = fit_power(rows, y1)
    plt.plot(rows, power_law(rows, a, b), 'b--', alpha=0.5, linewidth=4.0)

    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')

    plt.legend(["Random Sample $\mathbf{E}[W_2^2]$", r"Dist-NN, $N^{%.2f}$" % (b)], 
               borderpad=0.1, labelspacing=0)
    plt.xlabel("$N$")
    plt.ylabel("$W_2^2$ Error")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.4)
    ax.set_ylim(bottom=0, top=0.3)
    ax.set_xlim(left=0)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confidence_bands(est_dist, neighbor_dists, neighbor_emp_dists, true_dist_ppf,
                          n_samples, alpha=0.05, save_path=None):
    """
    Plot confidence bands for distribution estimates.
    
    Parameters
    ----------
    est_dist : np.ndarray
        Estimated distribution samples
    neighbor_dists : list
        List of true neighbor distributions (scipy.stats objects)
    neighbor_emp_dists : np.ndarray
        Empirical neighbor distributions
    true_dist_ppf : callable
        True distribution quantile function
    n_samples : int
        Number of samples per distribution
    alpha : float, optional
        Significance level (default: 0.05)
    save_path : str, optional
        Path to save the figure
    """
    from . import bootstrap
    
    x = np.linspace(0, 1, num=1000)
    steps = [i / (n_samples + 1) for i in range(1, n_samples + 1)]

    # Bonferroni correction
    alpha_corrected = alpha / n_samples
    z_alpha = scipy.stats.norm.ppf(1 - (alpha_corrected / 2))

    mpl.rcParams['legend.fontsize'] = 15

    plt.figure()
    plt.plot(x, np.quantile(est_dist, x), label='Dist-NN', linewidth=2)
    plt.plot(x, true_dist_ppf(x), label='True Distribution', linewidth=2, linestyle='--', color='k')
    plt.xlabel('$p$')
    plt.xlim([0, 1])
    plt.ylabel('$F^{-1}(p)$')

    print(f'Number of neighbors: {len(neighbor_dists)}')

    std_true = np.array([np.sqrt(bootstrap.get_variance_bootstrap_neighbors(s, n_samples, neighbor_dists)) 
                         for s in steps])
    top = est_dist + (z_alpha * std_true)
    bottom = est_dist - (z_alpha * std_true)
    plt.plot(x, np.quantile(bottom, x), color='black', linewidth=0.25, alpha=0.5)
    plt.plot(x, np.quantile(top, x), color='black', linewidth=0.25, alpha=0.5)
    plt.fill_between(x, np.quantile(bottom, x), np.quantile(top, x), 
                     alpha=0.4, 
                     color='tab:blue',
                     interpolate=False, 
                     label='95\% Asymptotic CB')

    std_bootstrap = np.sqrt(bootstrap.get_variance_bootstrap_everything(n_samples, neighbor_emp_dists, 
                                                              num_resamples_n=50,
                                                              num_resamples_neighbors=50))
    top = est_dist + (z_alpha * std_bootstrap)
    bottom = est_dist - (z_alpha * std_bootstrap)

    plt.plot(x, np.quantile(bottom, x), color='black', linewidth=0.25, alpha=0.5)
    plt.plot(x, np.quantile(top, x), color='black', linewidth=0.25, alpha=0.5)
    plt.fill_between(x, np.quantile(bottom, x), np.quantile(top, x), 
                     alpha=0.3, 
                     color='tab:orange',
                     interpolate=False, 
                     label='95\% Bootstrap CB')

    plt.legend()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(dist_errors_dict, save_path="figures/sim_all_metrics.pdf"):
    """
    Plot comparison of different metrics across methods.
    
    Parameters
    ----------
    dist_errors_dict : dict
        Dictionary containing error arrays for different methods and metrics
    save_path : str, optional
        Path to save the figure
    """
    label_size = 30
    mpl.rcParams['xtick.labelsize'] = label_size 
    mpl.rcParams['ytick.labelsize'] = label_size 
    mpl.rcParams['axes.labelsize'] = label_size
    mpl.rcParams['axes.titlesize'] = label_size
    mpl.rcParams['lines.markersize'] = 5
    mpl.rcParams['grid.linewidth'] = 2.5
    mpl.rcParams['legend.fontsize'] = label_size

    plt.figure(figsize=(20, 5))
    ax = sns.boxplot(data=[
        np.array(dist_errors_dict['mean_dnn']) * 100,
        np.array(dist_errors_dict['mean_rand']) * 100,
        np.array(dist_errors_dict['median_dnn']) * 100,
        np.array(dist_errors_dict['median_rand']) * 100,
        np.array(dist_errors_dict['std_dnn']) * 100,
        np.array(dist_errors_dict['std_rand']) * 100,
        np.array(dist_errors_dict['var_dnn']) * 100,
        np.array(dist_errors_dict['var_rand']) * 100,
    ])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.4)
    plt.ylim([0, 100])
    ax.set_xticklabels(["Mean\n(Dist-NN)", "Mean\n(Baseline)", "Median\n(Dist-NN)", 
                        "Median\n(Baseline)", "Std. Dev\n(Dist-NN)", "Std. Dev.\n(Baseline)", 
                        "VaR(5\%)\n(Dist-NN)", "VaR(5\%)\n(Baseline)"])
    plt.ylabel("Relative Error (\%)")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

