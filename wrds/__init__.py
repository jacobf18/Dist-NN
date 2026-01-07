"""
WRDS Distributional Nearest Neighbors Package

This package provides tools for downloading WRDS IBES data and performing
distributional nearest neighbors prediction for earnings forecasts.
"""

from .download_data import (
    connect_wrds,
    load_tickers_from_file,
    download_ibes_tickers,
    download_ibes_data,
    load_ibes_data,
    align_dates,
    create_quarterly_data,
    save_quarterly_data,
    load_quarterly_data,
    get_rows_cols
)

from .dist_nn import (
    wasserstein2,
    empirical_quantile_function,
    linear_combination,
    barycenter,
    expectation,
    var,
    squared_diff,
    relative_error,
    get_dist,
    get_avg_dist_trains,
    optimize_eta,
    get_similar_tickers,
    predict_distribution,
    test_train_wasserstein,
    get_earnings_time,
    get_doubly_robust_estimate,
    get_empirical_quantile_functions,
    optimize_eta_doubly_robust,
    train_test_cell,
    evaluate_eta_test
)

from .plots import (
    convert_to_hist,
    plot_histogram_comparison,
    plot_error_boxplots,
    plot_quantile_comparison,
    plot_individual_metric_boxplot
)

from .evaluation import (
    rolling_window_evaluation,
    growing_window_evaluation,
    seasonal_window_evaluation
)

from .experiments import (
    run_doubly_robust_experiment,
    run_item_item_evaluation,
    run_user_user_evaluation,
    calculate_avg_estimators
)

__version__ = '0.1.0'

