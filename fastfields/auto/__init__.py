"""fastfields.auto: a backend-dispatching interface over fastfields wrappers.

Every function here dispatches on the **type of the first array argument** and
forwards the call to the matching backend package:

* ``numpy.ndarray`` -> :mod:`fastfields.numpy`
* ``torch.Tensor``  -> :mod:`fastfields.torch`
* ``cupy.ndarray``  -> :mod:`fastfields.cupy`

Backends are imported **lazily**: only the backend for the array type
actually passed is imported, so ``fastfields.auto`` works with any subset
of the optional backends installed (only ``fastfields-helpers`` -- the
pure-Python enums/normalisers, no compiled extension -- is a hard
dependency; ``fastfields.auto`` never touches the compiled
``fastfields-dlpack`` bindings directly). A clear error is raised for an
unknown array type, and a clear ``ImportError`` if the required backend
package (or its own dependency, e.g. torch/cupy) is missing.

Every backend wrapper package exposes the same canonical ``dt_*`` / ``sym_*``
names, and this module exposes one unified name per operation that maps to each
backend's function. Arguments are forwarded unchanged, so each dispatched call
shares the signature of the selected backend's function.

The dispatchers are split by category into :mod:`._dt` (distance transforms),
:mod:`._sym` (compact-symmetric ops), :mod:`._resample` (resampling / spline
coefficients), :mod:`._pushpull` (spline gather/scatter) and :mod:`._reg`
(spatial regularisers), all built on the shared machinery in :mod:`._util`.

What is not dispatched here, and why
------------------------------------
This module is a *dispatcher*: it picks a backend from the type of the first
array argument and forwards every argument unchanged. Two kinds of operation
cannot be reached through it, and both are deliberate.

**Builders that take a shape, not an array.** ``field_diag``, ``flow_diag``,
``field_kernel`` and ``flow_kernel`` allocate their output from a rank and a
set of penalty weights -- there is no array to dispatch on, so there is
nothing for this module to decide. Call them on a concrete backend
(``fastfields.numpy.field_diag(...)``) and pass the result into the
dispatched operators. Their ``*_add``/``*_sub`` accumulate forms *do* take
an array and are dispatched.

**Adjoints that exist for autograd.** ``pull_backward``, ``push_backward``,
``count_backward`` and ``grad_backward`` are the gradients of the pushpull
operators; only :mod:`fastfields.torch` has an autograd engine, and there
they are wired into the forward op's ``autograd.Function`` rather than
exported. There is nothing uniform to dispatch, so the supported spelling is
``fastfields.auto.pull(...)`` followed by ``.backward()`` on a torch tensor,
or ``fastfields.dlpack.pull_backward(...)`` raw.

Operations that *some* backends have are dispatched with a partial table and
raise ``NotImplementedError`` naming the backends that do have them --
``field_cg`` (CPU-only in the C++ library, so numpy), ``field_precond_`` /
``flow_precond_`` (numpy and cupy), ``sym_matvec_backward`` (numpy and cupy).
Listing them with a clear failure is better than omitting them, which would
make the asymmetry invisible from here.

If a backend turns out not to have an operation this module says it does --
a mixed install, where ``fastfields.auto`` is newer than the wrapper -- the
call raises ``NotImplementedError`` saying which package to upgrade, not a
bare ``AttributeError`` from a module the caller never named.
"""

from __future__ import annotations

# Enums are backend-independent (they live in fastfields.helpers, the
# pure-Python dependency this package now uses -- see pyproject.toml).
from fastfields.helpers import Bound, Spline

from ._dt import (
    dt_euclidean,
    dt_euclidean_,
    dt_l1,
    dt_l1_,
    dt_mesh,
    dt_spline_brent,
    dt_spline_gaussnewton,
    dt_spline_table,
)
from ._pushpull import count, grad, pull, push
from ._reg import (
    field_adddiag,
    field_adddiag_,
    field_addkernel,
    field_addkernel_,
    field_addmatvec,
    field_addmatvec_,
    field_cg,
    field_diag_rls,
    field_forward,
    field_matvec,
    field_matvec_rls,
    field_precond,
    field_precond_,
    field_relax,
    field_relax_rls,
    field_subdiag,
    field_subdiag_,
    field_subkernel,
    field_subkernel_,
    field_submatvec,
    field_submatvec_,
    flow_adddiag,
    flow_adddiag_,
    flow_addkernel,
    flow_addkernel_,
    flow_addmatvec,
    flow_addmatvec_,
    flow_diag_rls,
    flow_forward,
    flow_matvec,
    flow_matvec_rls,
    flow_precond,
    flow_precond_,
    flow_relax,
    flow_relax_rls,
    flow_subdiag,
    flow_subdiag_,
    flow_subkernel,
    flow_subkernel_,
    flow_submatvec,
    flow_submatvec_,
)
from ._resample import resample, restriction, spline_coeff, spline_coeff_
from ._sym import (
    sym_addmatvec_,
    sym_invert,
    sym_invert_,
    sym_matvec,
    sym_matvec_backward,
    sym_solve,
    sym_solve_,
    sym_submatvec_,
)

__version__ = "0.1.0"

__all__ = [
    "Spline",
    "Bound",
    "__version__",
    # distance transforms
    "dt_euclidean",
    "dt_euclidean_",
    "dt_l1",
    "dt_l1_",
    "dt_spline_table",
    "dt_spline_brent",
    "dt_spline_gaussnewton",
    "dt_mesh",
    # compact-symmetric ops
    "sym_matvec",
    "sym_matvec_backward",
    "sym_addmatvec_",
    "sym_submatvec_",
    "sym_solve",
    "sym_solve_",
    "sym_invert",
    "sym_invert_",
    # resampling
    "resample",
    "restriction",
    "spline_coeff",
    "spline_coeff_",
    # pushpull
    "pull",
    "push",
    "count",
    "grad",
    # regularisers (operators; plain *_diag / flow_kernel are backend-specific)
    "field_matvec",
    "field_addmatvec",
    "field_addmatvec_",
    "field_submatvec",
    "field_submatvec_",
    "field_adddiag",
    "field_adddiag_",
    "field_subdiag",
    "field_subdiag_",
    "field_addkernel",
    "field_addkernel_",
    "field_subkernel",
    "field_subkernel_",
    "field_relax",
    "field_cg",
    "field_matvec_rls",
    "field_diag_rls",
    "field_relax_rls",
    "field_precond",
    "field_precond_",
    "field_forward",
    "flow_matvec",
    "flow_addmatvec",
    "flow_addmatvec_",
    "flow_submatvec",
    "flow_submatvec_",
    "flow_adddiag",
    "flow_adddiag_",
    "flow_subdiag",
    "flow_subdiag_",
    "flow_addkernel",
    "flow_addkernel_",
    "flow_subkernel",
    "flow_subkernel_",
    "flow_relax",
    "flow_matvec_rls",
    "flow_diag_rls",
    "flow_relax_rls",
    "flow_precond",
    "flow_precond_",
    "flow_forward",
]
