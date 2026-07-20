"""fastfields.any: a backend-dispatching interface over fastfields wrappers.

Every function here dispatches on the **type of the first array argument** and
forwards the call to the matching backend package:

* ``numpy.ndarray`` -> :mod:`fastfields.numpy`
* ``torch.Tensor``  -> :mod:`fastfields.torch`
* ``cupy.ndarray``  -> :mod:`fastfields.cupy`

Backends are imported **lazily**: only the backend for the array type actually
passed is imported, so ``fastfields.any`` works with any subset of the optional
backends installed (only ``fastfields-dlpack`` is a hard dependency). A clear
error is raised for an unknown array type, and a clear ``ImportError`` if the
required backend package (or its own dependency, e.g. torch/cupy) is missing.

Because the underlying wrapper packages use slightly different function names
(e.g. numpy's ``euclidean_distance_transform`` vs torch/cupy's
``dt_euclidean``, or numpy's ``mesh_distance`` vs torch/cupy's ``dt_mesh``),
this module exposes a single unified name per operation and maps it to each
backend's actual function. Arguments are forwarded unchanged, so each
dispatched call shares the signature of the selected backend's function.

The dispatchers are split by category into :mod:`._dt` (distance transforms),
:mod:`._sym` (compact-symmetric ops) and :mod:`._resample` (resampling / spline
coefficients), all built on the shared machinery in :mod:`._util`.
"""

from __future__ import annotations

# Enums are backend-independent (they live in fastfields.dlpack).
from fastfields.dlpack import Bound, Spline

from ._dt import (
    dt_euclidean,
    dt_l1,
    mesh_distance,
    spline_distance_brent,
    spline_distance_gaussnewton,
    spline_distance_table,
)
from ._resample import resample, restriction, spline_coeff
from ._sym import sym_invert, sym_matvec, sym_matvec_backward, sym_solve

__version__ = "0.1.0"

__all__ = [
    "Spline",
    "Bound",
    "__version__",
    # distance transforms
    "dt_euclidean",
    "dt_l1",
    "spline_distance_table",
    "spline_distance_brent",
    "spline_distance_gaussnewton",
    "mesh_distance",
    # compact-symmetric ops
    "sym_matvec",
    "sym_matvec_backward",
    "sym_solve",
    "sym_invert",
    # resampling
    "resample",
    "restriction",
    "spline_coeff",
]
