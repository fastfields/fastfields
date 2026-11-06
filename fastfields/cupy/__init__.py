"""CuPy interface to the fastfields operators, on the GPU.

The package mirrors `fastfields.numpy`, but its arrays are CuPy arrays living
in CUDA device memory. They reach the compiled library through DLPack, which
shares the device memory instead of copying it.

Functions take CuPy arrays and return newly allocated CuPy arrays. Inputs must
be float32 or float64 and are passed with their own strides, so nothing is
made contiguous; outputs are allocated contiguous. Functions whose name ends
in an underscore (`dt_euclidean_`, `sym_solve_` and so on) update their first
argument in place and return it.

All work is queued on CuPy's current CUDA stream: every wrapper forwards
`cupy.cuda.get_current_stream().ptr` to the library, so the kernels are
ordered with respect to the surrounding CuPy operations. To run on another
stream, call the function inside a `with my_stream:` block.

!!! note "Availability"
    Two things can be missing, and they are treated differently.

    CuPy itself is imported lazily, so `import fastfields.cupy` succeeds where
    CuPy is not installed or there is no GPU; a clear `ImportError` is raised
    only when a wrapper is called. That keeps the static parts of the package,
    the enums and the argument normalisers, importable and testable anywhere.

    The CUDA kernels are checked eagerly instead. `import fastfields.cupy`
    raises `ImportError` when the installed `fastfields-dlpack` is a CPU-only
    build, and the message names the wheel-index lane to install. Without the
    kernels nothing here can work, so there is no point in waiting for the
    first call.

!!! note "What this wrapper deliberately does not expose"
    Of the 58 public operations of the C++ hub, this package wraps all but
    five.

    **The adjoints of the sampling operators.** `pull_backward`,
    `push_backward`, `count_backward` and `grad_backward` give the gradient
    with respect to both the sampled field and the sampling coordinates. They
    exist so that an autograd engine can differentiate through `grid`, and
    only `fastfields.torch` has one. A wrapper here would be a function that
    nothing calls, with several pre-zeroed device buffers for the caller to
    get right. They remain available as `fastfields.dlpack.pull_backward(...)`
    and so on.

    **The conjugate-gradient solver.** `field_cg` is CPU-only in the C++
    library: its dot products need a device-side reduction that the CUDA
    backend does not implement yet (fastfields-lib#34), and the hub rejects
    CUDA arrays instead of quietly running on the host. A
    `fastfields.cupy.field_cg` could therefore only ever raise.
    `field_relax` and `flow_relax` run on CUDA and solve the same system by
    Gauss-Seidel sweeps.

    `tests/test_api_coverage_cupy.py` fails if the set of omitted operations
    ever changes by accident.
"""

from __future__ import annotations

# Spline / Bound are plain IntEnums in fastfields.helpers (no cupy needed).
# `__version__` is the whole `fastfields` distribution's version, read from
# the single `_version.py` that versioningit writes into fastfields.helpers;
# the five merged subpackages all report the same string. See pyproject.toml.
from fastfields.helpers import Bound, Spline, __version__

# Import-time CUDA-build guard; first local import on purpose, so the failure
# happens at `import fastfields.cupy` and not inside the first wrapper call.
# See _backend.py for why this namespace refuses to load on a CPU-only build
# while fastfields.torch imports cleanly and checks lazily.
from . import _backend  # noqa: F401
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
    field_diag,
    field_diag_rls,
    field_forward,
    field_kernel,
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
    flow_diag,
    flow_diag_rls,
    flow_forward,
    flow_kernel,
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
from ._util import current_stream_ptr

__all__ = [
    # distance transforms
    "dt_euclidean",
    "dt_euclidean_",
    "dt_l1",
    "dt_l1_",
    "dt_spline_table",
    "dt_spline_brent",
    "dt_spline_gaussnewton",
    "dt_mesh",
    # symmetric-matrix ops
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
    # regularisers
    "field_matvec",
    "field_addmatvec",
    "field_addmatvec_",
    "field_submatvec",
    "field_submatvec_",
    "field_diag",
    "field_adddiag",
    "field_adddiag_",
    "field_subdiag",
    "field_subdiag_",
    "field_kernel",
    "field_addkernel",
    "field_addkernel_",
    "field_subkernel",
    "field_subkernel_",
    "field_relax",
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
    "flow_diag",
    "flow_adddiag",
    "flow_adddiag_",
    "flow_subdiag",
    "flow_subdiag_",
    "flow_kernel",
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
    # enums / helpers
    "Spline",
    "Bound",
    "current_stream_ptr",
    "__version__",
]
