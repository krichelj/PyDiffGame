"""Shared type aliases for :mod:`PyDiffGame`.

Kept in one place so the rest of the package can use precise, readable
annotations without repeating ``numpy.typing`` boilerplate.  The aliases are
plain assignments (rather than :pep:`695` ``type`` statements) so the package
keeps importing on Python 3.11, while still reading cleanly on 3.14.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Union

import numpy as np
from numpy.typing import NDArray

#: A real-valued, floating point numpy array (vector or matrix).
FloatArray = NDArray[np.float64]

#: Anything that can be coerced into a float array (lists, tuples, scalars...).
ArrayLike = Union[FloatArray, Sequence[float], Sequence[Sequence[float]], float, int]

#: A filesystem path, accepted either as ``str`` or :class:`os.PathLike`.
PathInput = Union[str, "PathLike[str]"]

__all__ = ["FloatArray", "ArrayLike", "PathInput"]
