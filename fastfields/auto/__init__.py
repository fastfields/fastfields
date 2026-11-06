"""Backend-dispatching interface to the fastfields operators.

Every function here looks at the type of its first array argument and forwards
the call, unchanged, to the function of the same name in the matching backend:

- `numpy.ndarray`: `fastfields.numpy`
- `torch.Tensor`: `fastfields.torch`
- `cupy.ndarray`: `fastfields.cupy`

Arguments and results pass through untouched, so a dispatched call has the
signature of the backend function it selects. The reference page for each
operation is the one of its backend; this page documents what is specific to
dispatching.

Backends are imported lazily, and only when an array of their type is passed.
The package therefore works with any subset of them installed. Its only hard
dependency is `fastfields.helpers`; apart from `dlpack_backend`, which imports
the compiled `fastfields-dlpack` bindings when it is called, it does not touch
them. An object of any other type raises
`TypeError`, and a backend whose package is missing raises `ImportError`
naming what to install.

This namespace does not need any particular compute backend. `dlpack_backend`
reports which one is installed (`"cpu"`, `"cu126"` and so on). By contrast,
`fastfields.cupy` refuses to import on a CPU-only build and `fastfields.torch`
refuses a CUDA tensor on one.

!!! note "What is not dispatched here, and why"
    Two kinds of operation cannot be reached through this module, and both
    are deliberate.

    **Builders that take a shape, not an array.** `field_diag`, `flow_diag`,
    `field_kernel` and `flow_kernel` allocate their output from a rank and a
    set of penalty weights, so there is no array to dispatch on. Call them on
    a concrete backend, for example `fastfields.numpy.field_diag(...)`, and
    pass the result to the dispatched operators. The `*_add*` and `*_sub*`
    accumulate forms take an array and are dispatched.

    **Adjoints that exist for autograd.** `pull_backward`, `push_backward`,
    `count_backward` and `grad_backward` are the gradients of the sampling
    operators. Only PyTorch has an autograd engine, and there they are wired
    into the forward operation instead of being exported. The supported
    spelling is `fastfields.auto.pull(...)` followed by `.backward()` on a
    torch tensor, or `fastfields.dlpack.pull_backward(...)` for the raw
    function.

    Some operations exist on only some backends. They are dispatched with a
    partial table, and calling one on another backend raises
    `NotImplementedError` naming the backends that have it: `field_cg`
    (numpy only, because it is CPU-only in the C++ library),
    `field_precond_` and `flow_precond_` (numpy and cupy) and
    `sym_matvec_backward` (numpy and cupy). Listing them with a clear failure
    keeps the asymmetry visible.

    If a backend lacks an operation that this module says it has, which
    points to a stale or partial installation, the call raises
    `NotImplementedError` saying so, not a bare `AttributeError` from a module
    the caller never named.
"""

from __future__ import annotations

# Enums are backend-independent (they live in fastfields.helpers, the
# pure-Python dependency this package now uses -- see pyproject.toml).
# `__version__` is the whole `fastfields` distribution's version, read from
# the single `_version.py` that versioningit writes into fastfields.helpers;
# the five merged subpackages all report the same string. See pyproject.toml.
from fastfields.helpers import Bound, Spline, __version__

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
from ._util import dlpack_backend

__all__ = [
    "Spline",
    "Bound",
    "__version__",
    "dlpack_backend",
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
