"""
Example script demonstrating how to run experiments from the notebook using the wrds package.

This script shows how to:
1. Run doubly-robust estimator experiments
2. Run evaluation experiments with different windowing strategies
3. Run item-item and user-user evaluations
"""

import pandas as pd
import numpy as np
from wrds import (
    load_quarterly_data,
    get_rows_cols,
    run_doubly_robust_experiment,
    run_item_item_evaluation,
    run_user_user_evaluation,
    rolling_window_evaluation,
    growing_window_evaluation,
    seasonal_window_evaluation,
    calculate_avg_estimators
)

# Load the quarterly data
print("Loading quarterly data...")
quarterly_data, quarterly_actual, quarterly_means = load_quarterly_data()

# Example 1: Run doubly-robust experiment
print("\n=== Example 1: Doubly-Robust Experiment ===")
dates = [
    (2018, 1), (2018, 2), (2018, 3), (2018, 4),
    (2019, 1), (2019, 2), (2019, 3), (2019, 4), 
    (2020, 1), (2020, 2), (2020, 3), (2020, 4),
    (2021, 1), (2021, 2), (2021, 3), (2021, 4),
    (2022, 1), (2022, 2), (2022, 3), (2022, 4),
    (2023, 1), (2023, 2), (2023, 3), (2023, 4),
    (2024, 1), (2024, 2), (2024, 3), (2024, 4)
]

# Get list of tickers from quarterly_data
tickers = list(set([k[0] for k in quarterly_data.keys() if k[0] is not None]))[:10]  # Use first 10 for demo

# Run experiment (commented out to avoid long execution)
# results_dr = run_doubly_robust_experiment(
#     tickers=tickers,
#     dates=dates,
#     quarterly_data=quarterly_data,
#     quarterly_actual=quarterly_actual,
#     output_file='doubly_robust_errors.csv',
#     dates_per_ticker=3,
#     verbose=True
# )
# print(f"Doubly-robust experiment completed. Results saved to doubly_robust_errors.csv")
# print(results_dr.head())


# Example 2: Run item-item evaluation
print("\n=== Example 2: Item-Item Evaluation ===")
# Get tickers for item-item evaluation
_, cols = get_rows_cols(user_user=False)
subset_tickers = list(cols[:20])  # Use first 20 tickers for demo

# Run evaluation (commented out to avoid long execution)
# results_item = run_item_item_evaluation(
#     tickers=subset_tickers,
#     year=2024,
#     quarter=3,
#     quarterly_data=quarterly_data,
#     quarterly_actual=quarterly_actual,
#     quarterly_means=quarterly_means,
#     output_file='errors_item_item_small.csv',
#     verbose=True
# )
# print(f"Item-item evaluation completed. Results saved to errors_item_item_small.csv")
# print(results_item.head())


# Example 3: Run user-user evaluation
print("\n=== Example 3: User-User Evaluation ===")
_, cols = get_rows_cols(user_user=True)
train_test_split = 6
start = 12
train_cols = cols[start:train_test_split + start]
test_cols = cols[start + train_test_split:start + train_test_split + 1]

# Get subset of tickers
subset_tickers = list(set([k[0] for k in quarterly_data.keys() if k[0] is not None]))[:10]

# Run evaluation (commented out to avoid long execution)
# results_user = run_user_user_evaluation(
#     tickers=subset_tickers,
#     train_cols=train_cols,
#     test_cols=test_cols,
#     quarterly_data=quarterly_data,
#     quarterly_actual=quarterly_actual,
#     quarterly_means=quarterly_means,
#     output_file='test_errors_4_periods.csv',
#     verbose=True
# )
# print(f"User-user evaluation completed. Results saved to test_errors_4_periods.csv")
# print(results_user.head())


# Example 4: Rolling window evaluation
print("\n=== Example 4: Rolling Window Evaluation ===")
subset_tickers = list(set([k[0] for k in quarterly_data.keys() if k[0] is not None]))[:5]  # Use 5 for demo

# Run evaluation (commented out to avoid long execution)
# results_rolling = rolling_window_evaluation(
#     tickers=subset_tickers,
#     quarterly_data=quarterly_data,
#     quarterly_actual=quarterly_actual,
#     quarterly_means=quarterly_means,
#     window_size=10,
#     start_idx=10,
#     output_file='rolling_window_10.csv',
#     user_user=True
# )
# print(f"Rolling window evaluation completed. Results saved to rolling_window_10.csv")
# print(results_rolling.head())


# Example 5: Growing window evaluation
print("\n=== Example 5: Growing Window Evaluation ===")
# Run evaluation (commented out to avoid long execution)
# results_growing = growing_window_evaluation(
#     tickers=subset_tickers,
#     quarterly_data=quarterly_data,
#     quarterly_actual=quarterly_actual,
#     quarterly_means=quarterly_means,
#     start_idx=10,
#     output_file='growing_window.csv',
#     user_user=True
# )
# print(f"Growing window evaluation completed. Results saved to growing_window.csv")
# print(results_growing.head())


# Example 6: Seasonal window evaluation
print("\n=== Example 6: Seasonal Window Evaluation ===")
# Run evaluation (commented out to avoid long execution)
# results_seasonal = seasonal_window_evaluation(
#     tickers=subset_tickers,
#     quarterly_data=quarterly_data,
#     quarterly_actual=quarterly_actual,
#     quarterly_means=quarterly_means,
#     train_years=[2020, 2021, 2022, 2023],
#     test_year=2024,
#     output_file='seasonal_test.csv',
#     user_user=True
# )
# print(f"Seasonal window evaluation completed. Results saved to seasonal_test.csv")
# print(results_seasonal.head())


# Example 7: Calculate average estimators
print("\n=== Example 7: Calculate Average Estimators ===")
avg_estimators = calculate_avg_estimators(
    tickers=subset_tickers,
    quarterly_data=quarterly_data
)
print("Average number of estimators per ticker:")
for ticker, avg in list(avg_estimators.items())[:5]:
    print(f"  {ticker}: {avg:.2f}")

print("\n=== Examples completed! ===")
print("Uncomment the experiment calls above to run them.")

