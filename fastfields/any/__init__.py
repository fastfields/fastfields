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

Every backend wrapper package exposes the same canonical ``dt_*`` / ``sym_*``
names, and this module exposes one unified name per operation that maps to each
backend's function. Arguments are forwarded unchanged, so each dispatched call
shares the signature of the selected backend's function.

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
    dt_mesh,
    dt_spline_brent,
    dt_spline_gaussnewton,
    dt_spline_table,
)
from ._pushpull import count, grad, pull, push
from ._reg import (
    field_matvec,
    flow_diag_add,
    flow_diag_add_,
    flow_diag_sub,
    flow_diag_sub_,
    flow_forward,
    flow_matvec,
    flow_matvec_add,
    flow_matvec_add_,
    flow_matvec_sub,
    flow_matvec_sub_,
    flow_precond,
    flow_relax,
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
    "dt_spline_table",
    "dt_spline_brent",
    "dt_spline_gaussnewton",
    "dt_mesh",
    # compact-symmetric ops
    "sym_matvec",
    "sym_matvec_backward",
    "sym_solve",
    "sym_invert",
    # resampling
    "resample",
    "restriction",
    "spline_coeff",
    # pushpull
    "pull",
    "push",
    "count",
    "grad",
    # regularisers (operators; plain *_diag / flow_kernel are backend-specific)
    "field_matvec",
    "flow_matvec",
    "flow_matvec_add",
    "flow_matvec_add_",
    "flow_matvec_sub",
    "flow_matvec_sub_",
    "flow_diag_add",
    "flow_diag_add_",
    "flow_diag_sub",
    "flow_diag_sub_",
    "flow_relax",
    "flow_precond",
    "flow_forward",
]
