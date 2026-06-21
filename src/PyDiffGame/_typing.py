"""Shared type aliases for :mod:`PyDiffGame`.

Kept in one place so the rest of the package can use precise, readable
annotations without repeating ``numpy.typing`` boilerplate.  The aliases are
plain assignments (rather than :pep:`695` ``type`` statements) so the package
keeps importing on Python 3.11, while still reading cleanly on 3.14.
"""

from collections.abc import Sequence
from os import PathLike
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: A real-valued, floating point numpy array (vector or matrix).
FloatArray = NDArray[np.floating[Any]]

#: Anything that can be coerced into a float array (lists, tuples, scalars...).
ArrayLike = FloatArray | Sequence[float] | Sequence[Sequence[float]] | float | int

#: A filesystem path, accepted either as ``str`` or :class:`os.PathLike`.
PathInput = str | PathLike[str]

__all__ = ["FloatArray", "ArrayLike", "PathInput"]
