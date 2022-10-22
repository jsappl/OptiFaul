"""A collection of helper functions."""

from typing import TYPE_CHECKING, Type

import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray


def namestr_from(_class: "Type") -> str:
    """Extract name string from class instance."""
    return _class.__class__.__name__


def r_squared(targets: "ndarray", predictions: "ndarray") -> float:
    """Calculate the coefficient of determination.

    Args:
        targets: The ground truth.
        predictions: The model output.

    Returns:
        The coefficient of determination, also called R^2.
    """
    idx = np.squeeze(np.argwhere(~np.isnan(predictions)))
    corr_matrix = np.corrcoef(targets[idx], predictions[idx])
    corr = corr_matrix[0, 1]
    return corr**2
