"""NumPy interface to the fastfields operators.

Every function takes NumPy arrays and returns newly allocated NumPy arrays; its
inputs are never modified. The exceptions are the functions whose name ends in
an underscore, such as `dt_euclidean_`, `sym_solve_` and `spline_coeff_`: they
update their first argument in place and return it. `fastfields.cupy` follows
the same rule, while `fastfields.torch` leaves out the in-place spelling of
operations whose backward pass would need the value being overwritten (see
`API_CONTRACT.md`, "In-place policy").

The operations fall into five groups: distance transforms (`dt_*`), linear
algebra on fields of small symmetric matrices (`sym_*`), spline resampling and
prefiltering (`resample`, `restriction`, `spline_coeff`), spline sampling
(`pull`, `push`, `count`, `grad`) and spatial regularisers (`field_*` and
`flow_*`). A *field* is a multi-channel image with any number of channels. A
*flow* is a vector field with one component per spatial dimension.

**Batch axes are broadcast.** The compiled library wants every input of an
operation to have the same batch shape, so the wrappers first expand each
input to the common one. The expansion gives the broadcast axes a stride of
zero, which makes it a view of the original memory: nothing is copied. Outputs
carry the broadcast batch shape.

**Inputs are not copied if it can be avoided.** They are passed with their own
strides, contiguous or not. The exceptions are read-only arrays, which NumPy
cannot export through DLPack and which are therefore copied, and arrays of a
dtype other than float32 or float64, which are converted to float64. The
in-place functions never convert; they raise `TypeError` instead.

**Spline orders and boundary conditions** are chosen with `order` and `bound`.
Either accepts an integer, a `Spline` or `Bound` member, or a name such as
`"cubic"` or `"dct2"`.

!!! note "What this wrapper deliberately does not expose"
    Of the 58 public operations of the C++ hub, this package wraps all but
    four: `pull_backward`, `push_backward`, `count_backward` and
    `grad_backward`. They are the adjoints of the sampling operators with
    respect to both the sampled field and the sampling coordinates, and they
    exist so that an autograd engine can differentiate through `grid`. Only
    `fastfields.torch` has one. A wrapper here would be a function that
    nothing calls, with several pre-zeroed buffers for the caller to get
    right. They remain available through the raw bindings, as
    `fastfields.dlpack.pull_backward(...)` and so on.

    `tests/test_api_coverage_numpy.py` fails if the set of omitted operations
    ever changes by accident, and `tools/api_inventory.py` in
    `fastfields-dlpack` prints the full matrix.
"""

from __future__ import annotations

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
    CGInfo,
    field_adddiag,
    field_adddiag_,
    field_addkernel,
    field_addkernel_,
    field_addmatvec,
    field_addmatvec_,
    field_cg,
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
    sym_channels_from_packed,
    sym_invert,
    sym_invert_,
    sym_matvec,
    sym_matvec_backward,
    sym_solve,
    sym_solve_,
    sym_submatvec_,
)

__all__ = [
    "__version__",
    "Spline",
    "Bound",
    "dt_euclidean",
    "dt_euclidean_",
    "dt_l1",
    "dt_l1_",
    "sym_matvec",
    "sym_matvec_backward",
    "sym_addmatvec_",
    "sym_submatvec_",
    "sym_solve",
    "sym_solve_",
    "sym_invert",
    "sym_invert_",
    "resample",
    "restriction",
    "spline_coeff",
    "spline_coeff_",
    "dt_spline_table",
    "dt_spline_brent",
    "dt_spline_gaussnewton",
    "dt_mesh",
    "sym_channels_from_packed",
    "pull",
    "push",
    "count",
    "grad",
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
    "field_cg",
    "CGInfo",
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
]
