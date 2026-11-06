"""PyTorch interface to the fastfields operators, with autograd.

Functions take tensors and return newly allocated tensors. Most of them are
differentiable, so they can sit inside a model and gradients flow through them.
Functions whose name ends in an underscore update their first argument in place
and return it. CPU tensors work with any build of `fastfields-dlpack`. A CUDA
tensor needs a CUDA build; handing one to a CPU-only build raises
`RuntimeError` at the call, naming the wheel-index lane to install.

**Differentiable operations.** Gradients are available with respect to the
following arguments.

- `sym_matvec`: `mat` and `vec`.
- `sym_solve` and `sym_solve_`: `vec` (the right-hand side) only.
- `resample`, `restriction`, `spline_coeff` and `spline_coeff_`: the input.
- `pull`, `push` and `grad`: the field and the sampling `grid`. `count`:
  `grid`. `grad` is itself a derivative operator, so differentiating it gives
  the second-order terms that an image gradient computed inside a loss needs.
- The regulariser operators: `field_matvec` and `flow_matvec` with respect to
  the input; their accumulate forms (`*_add*` and `*_sub*` of the product, the
  diagonal and the stencil) with respect to the accumulator and, for the
  product, the array it is applied to; and `field_forward`, `field_precond`
  and their flow twins, as described in each function.

**Operations that are not differentiable** are still exposed, for parity with
`fastfields.numpy` and `fastfields.cupy`. Their forward pass always runs, even
when an input requires grad; only calling `.backward()` through the output
raises `RuntimeError`, naming the operation. They are the distance transforms
(`dt_*`), `sym_invert` and `sym_invert_`, the iterative solvers `field_relax`,
`flow_relax` and their `*_relax_rls` variants, the stencil and diagonal
builders, and the gradients of `sym_addmatvec_` and `sym_submatvec_` with
respect to `mat` and `vec`.

!!! note "What this wrapper deliberately does not expose"
    Of the 58 public operations of the C++ hub, this package wraps all but
    eight, in two groups.

    **Adjoints, used but not re-exported.** `pull_backward`, `push_backward`,
    `count_backward`, `grad_backward` and `sym_matvec_backward` are the
    gradients behind the `autograd.Function` of each forward operation. The
    supported way to reach them is to call `pull`, `push`, `count`, `grad` or
    `sym_matvec` and then `.backward()`. Re-exporting them would offer a
    second, unchecked spelling of a gradient, with pre-zeroed buffers and an
    accumulate-or-overwrite contract for the caller to get right.
    `fastfields.dlpack` has them raw.

    **Solver entry points that need an autograd decision first.**
    `field_precond_`, `flow_precond_` and `field_cg`. The in-place
    preconditioner solves would have to settle how autograd treats a mutated
    buffer (`API_CONTRACT.md`, "In-place policy"), and `field_cg` runs an
    iterative solver whose gradient is a solve with the same operator. Both
    are worth doing and neither is a mechanical wrapper. `field_relax` is the
    in-place solver available today, and `fastfields.numpy` has all three for
    the case without autograd.

    `tests/test_api_coverage_torch.py` fails if the set of omitted operations
    ever changes by accident.
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
    sym_solve,
    sym_solve_,
    sym_submatvec_,
)

__all__ = [
    "__version__",
    "sym_matvec",
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
    "dt_euclidean",
    "dt_euclidean_",
    "dt_l1",
    "dt_l1_",
    "dt_mesh",
    "dt_spline_table",
    "dt_spline_brent",
    "dt_spline_gaussnewton",
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
    "field_matvec_rls",
    "field_diag_rls",
    "field_relax_rls",
    "field_precond",
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
    "flow_forward",
    "Spline",
    "Bound",
]
