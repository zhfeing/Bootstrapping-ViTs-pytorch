"""
Copied from https://github.com/deep-spin/entmax
"""

from .activations import sparsemax, entmax15, Sparsemax, Entmax15
from .root_finding import (
    sparsemax_bisect,
    entmax_bisect,
    SparsemaxBisect,
    EntmaxBisect,
)
from .losses import (
    sparsemax_loss,
    entmax15_loss,
    sparsemax_bisect_loss,
    entmax_bisect_loss,
    SparsemaxLoss,
    SparsemaxBisectLoss,
    Entmax15Loss,
    EntmaxBisectLoss,
)
