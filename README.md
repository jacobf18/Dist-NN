# Distributional Matrix Completion via Nearest Neighbors in the Wasserstein Space

Authors: Jacob Feitelberg, Kyuseong Choi, Anish Agarwal, Raaz Dwivedi

Distributional Nearest Neighbors is a framework for solving the distributional matrix completion problem. This is similar to scalar matrix completion, but each entry is a distribution of real numbers instead of a single scalar. We use the Wasserstein metric (in particular, the 2-Wasserstein metric) to determine how close distributions are to each other. We choose to use the Wasserstein metric because it retains geometric information when finding an average in the Wasserstein space. For instance, an average of multiple Gaussian distributions is again Gaussian.

## Installation

Install the package using pip:

```bash
pip install -e .
```

Or install in development mode with optional dependencies:

```bash
pip install -e ".[dev]"
```

This will install the `nn_dist_completion` package along with all required dependencies (numpy, scipy, matplotlib, seaborn, pandas, scikit-learn, hyperopt, tqdm).

For more detailed usage instructions, see the [package README](nn_dist_completion/README.md).
