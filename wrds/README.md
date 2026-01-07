# WRDS Distributional Nearest Neighbors (Dist-NN)

This package implements Distributional Nearest Neighbors for earnings forecast prediction using WRDS IBES data. The method uses Wasserstein distances to find similar companies/quarters and predicts distributions using barycenters of neighbor distributions.

## Overview

The package is organized into two main modules:

1. **`download_data.py`**: Handles downloading and processing WRDS IBES earnings data
2. **`dist_nn.py`**: Implements the Distributional Nearest Neighbors algorithm

## Installation

Required packages:
```bash
pip install wrds pandas numpy scipy matplotlib seaborn tqdm pickle5 hyperopt POT
```

## Module Descriptions

### `download_data.py`

This module handles all data downloading and preprocessing tasks:

#### Key Functions:

- **`connect_wrds(wrds_username)`**: Connect to WRDS database
- **`download_ibes_tickers(conn, us_tickers, ...)`**: Download IBES ticker mappings for US companies
- **`download_ibes_data(conn, ibes_tickers, ...)`**: Download IBES earnings forecast data and save to CSV files
- **`load_ibes_data(data_dir, ...)`**: Load IBES data from CSV files
- **`align_dates(ibes_data, base_ticker='AAPL')`**: Align announcement dates across companies using linear assignment (uses AAPL as reference)
- **`create_quarterly_data(ibes_data, aligned_dates, ...)`**: Organize data into quarterly structures
- **`save_quarterly_data(...)`** / **`load_quarterly_data(...)`**: Save/load processed quarterly data

#### Usage Example:

```python
from download_data import *

# Connect to WRDS
conn = connect_wrds('your_username')

# Load tickers
us_tickers = load_tickers_from_file('us_tickers.txt')

# Download IBES ticker mappings
ibes_tickers = download_ibes_tickers(conn, us_tickers[:2000])

# Download data
download_ibes_data(conn, ibes_tickers, output_dir='data/ibes')

# Load and process data
ibes_data = load_ibes_data('data/ibes', ibes_tickers)
aligned_dates = align_dates(ibes_data, base_ticker='AAPL')
quarterly_data, quarterly_actual, quarterly_means = create_quarterly_data(
    ibes_data, aligned_dates, starting_year=2010
)

# Save processed data
save_quarterly_data(quarterly_data, quarterly_actual, quarterly_means)
```

### `dist_nn.py`

This module implements the Distributional Nearest Neighbors algorithm:

#### Key Functions:

- **`wasserstein2(sample1, sample2)`**: Compute squared 2-Wasserstein distance between two samples
- **`empirical_quantile_function(samples)`**: Create empirical quantile function from sorted samples
- **`barycenter(quantile_fns)`**: Compute barycenter (average) of quantile functions
- **`expectation(quantile_func)`**: Compute expectation (mean) from quantile function
- **`var(quantile_func)`**: Compute variance from quantile function
- **`get_dist(ticker, year, quarter, ...)`**: Compute Wasserstein distances from a cell to neighbors
- **`get_similar_tickers(ticker, year, quarter, ...)`**: Find similar tickers based on distributional distances
- **`optimize_eta(...)`**: Optimize threshold parameter for selecting neighbors
- **`predict_distribution(...)`**: Predict distribution using Dist-NN
- **`test_train_wasserstein(...)`**: Test Dist-NN on missing data scenarios

#### Usage Example:

```python
from dist_nn import *
import pandas as pd

# Load quarterly data
from download_data import load_quarterly_data
quarterly_data, quarterly_actual, quarterly_means = load_quarterly_data()

# Set cutoff date
cutoff_date = pd.to_datetime('2019-11-01')

# Test on a specific cell
test_cell = ('AAPL', 2020, 1)
results = test_train_wasserstein(
    test_cell, cutoff_date, quarterly_data, quarterly_actual,
    user_user=False, verbose=True, num_missing=30
)

# Get predictions
for ticker, result in results.items():
    b_fn = result['b_fn']  # Predicted quantile function
    actual_data = result['actual_data']  # Actual forecast data
    
    # Compute statistics
    pred_mean = expectation(b_fn)
    pred_median = b_fn(np.array([0.5]))[0]
    pred_std = np.sqrt(var(b_fn))
    
    print(f"{ticker}: Mean={pred_mean:.4f}, Median={pred_median:.4f}, Std={pred_std:.4f}")
```

## Algorithm Overview

The Distributional Nearest Neighbors (Dist-NN) method works as follows:

1. **Distance Computation**: For each cell (ticker, year, quarter), compute Wasserstein distances to other cells in the same row/column
2. **Similarity Finding**: Find similar tickers/quarters based on distributional distance vectors
3. **Hyperparameter Optimization**: Optimize the distance threshold `eta` using training data
4. **Neighbor Selection**: Select neighbors within the threshold distance
5. **Distribution Prediction**: Compute the barycenter (average) of neighbor distributions as the prediction

## Data Structure

The processed data is organized as:

- **`quarterly_data`**: Dictionary mapping `(ticker, year, quarter)` to DataFrame with columns:
  - `value`: Forecast EPS values
  - `ann_datetime`: Announcement datetime for each forecast

- **`quarterly_actual`**: Dictionary mapping `(ticker, year, quarter)` to tuple:
  - `(actual_value, announcement_date, announcement_time)`

- **`quarterly_means`**: Dictionary mapping `(ticker, year, quarter)` to mean forecast value

## Additional Modules

### `evaluation.py`

This module provides functions for evaluating Dist-NN using different windowing strategies:

- **`rolling_window_evaluation()`**: Evaluate using a rolling window of fixed size
- **`growing_window_evaluation()`**: Evaluate using a growing window (all previous quarters)
- **`seasonal_window_evaluation()`**: Evaluate using seasonal windows (same quarter, different years)

### `experiments.py`

This module provides functions for running various experiments:

- **`run_doubly_robust_experiment()`**: Run doubly-robust estimator experiments on multiple tickers and dates
- **`run_item_item_evaluation()`**: Run item-item mode evaluation on a set of tickers
- **`run_user_user_evaluation()`**: Run user-user mode evaluation on a set of tickers
- **`calculate_avg_estimators()`**: Calculate average number of estimators per ticker

### Extended `dist_nn.py` Functions

Additional functions added for doubly-robust estimation and evaluation:

- **`get_earnings_time()`**: Get earnings announcement datetime for a cell
- **`get_doubly_robust_estimate()`**: Compute doubly-robust estimate combining user-user, item-item, and cross terms
- **`get_empirical_quantile_functions()`**: Calculate empirical quantile functions for doubly-robust estimation
- **`optimize_eta_doubly_robust()`**: Optimize both user-user and item-item thresholds simultaneously
- **`train_test_cell()`**: Train and test doubly-robust estimator on a single test cell
- **`evaluate_eta_test()`**: Evaluate a given eta threshold on test columns

## Example Usage

See `example_experiments.py` for complete examples of running experiments.

### Running Doubly-Robust Experiments

```python
from wrds import load_quarterly_data, run_doubly_robust_experiment

quarterly_data, quarterly_actual, quarterly_means = load_quarterly_data()
tickers = ['AAPL', 'MSFT', 'GOOG']
dates = [(2020, 1), (2020, 2), (2021, 1), (2021, 2)]

results = run_doubly_robust_experiment(
    tickers=tickers,
    dates=dates,
    quarterly_data=quarterly_data,
    quarterly_actual=quarterly_actual,
    output_file='doubly_robust_errors.csv'
)
```

### Running Window Evaluations

```python
from wrds import rolling_window_evaluation, seasonal_window_evaluation

# Rolling window
results = rolling_window_evaluation(
    tickers=tickers,
    quarterly_data=quarterly_data,
    quarterly_actual=quarterly_actual,
    quarterly_means=quarterly_means,
    window_size=10,
    output_file='rolling_window_10.csv'
)

# Seasonal window
results = seasonal_window_evaluation(
    tickers=tickers,
    quarterly_data=quarterly_data,
    quarterly_actual=quarterly_actual,
    quarterly_means=quarterly_means,
    train_years=[2020, 2021, 2022, 2023],
    test_year=2024,
    output_file='seasonal_test.csv'
)
```

## Notes

- The date alignment uses AAPL as the base ticker since it has complete data (20 quarters)
- The method filters data based on cutoff dates to avoid look-ahead bias
- Hyperparameter optimization uses Bayesian optimization (hyperopt) to find optimal distance thresholds
- The algorithm supports both "user-user" (same quarter, different tickers) and "item-item" (same ticker, different quarters) modes
- The doubly-robust estimator combines user-user and item-item predictions to potentially improve accuracy

## References

This implementation is based on research using Wasserstein distances for distributional completion in earnings forecast matrices.

