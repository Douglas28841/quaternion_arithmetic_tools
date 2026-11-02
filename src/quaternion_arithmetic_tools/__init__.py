"""Quaternion arithmetic tools.

Exports the core functions for working with four-component real quaternions
represented as tuples ``(w, x, y, z)``.
"""

from .core import (
    conjugate,
    hamilton_product,
    norm,
    normalize,
    rotate_vector,
)

__all__ = [
    "conjugate",
    "hamilton_product",
    "norm",
    "normalize",
    "rotate_vector",
]
