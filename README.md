# Dist-NN

Authors: Jacob Feitelberg, Anish Agarwal

Distributional Nearest Neighbors is a framework for solving the distributional table completion problem. This is similar to matrix completion, but each entry is a distribution of rela numbers instead of a single scalar. We use the Wasserstein metric (in particular, the 2-Wasserstein metric) to determine how close distributions are to each other.