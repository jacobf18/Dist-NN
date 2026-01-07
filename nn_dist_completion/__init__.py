"""
Nearest Neighbors Distribution Completion Package

This package provides tools for distributional matrix completion using
Wasserstein nearest neighbors.
"""

from . import utils
from . import nearest_neighbors
from . import simulation
from . import plotting
from . import config
from . import bootstrap

__all__ = ['utils', 'nearest_neighbors', 'simulation', 'plotting', 'config', 'bootstrap']

