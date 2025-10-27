"""
sklearn_from_scratch: Re-implementation of core scikit-learn algorithms

This package provides from-scratch implementations of popular machine learning
algorithms to help understand how they work under the hood.
"""

__version__ = '0.1.0'

from . import linear_models
from . import neighbors
from . import tree
from . import metrics
from . import preprocessing

__all__ = [
    'linear_models',
    'neighbors', 
    'tree',
    'metrics',
    'preprocessing',
]
