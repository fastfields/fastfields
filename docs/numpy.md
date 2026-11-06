# fastfields.numpy

`fastfields.numpy` runs the fastfields operators on NumPy arrays, on the CPU.
Arrays go in and freshly allocated arrays come out. Your inputs are left alone
unless you call one of the in-place variants, whose names end in an
underscore.

## Install

```sh
pip install "fastfields[numpy]" \
    --extra-index-url https://fastfields.github.io/whl/cpu/
```

The `numpy` extra installs NumPy itself. The `--extra-index-url` is what
selects the compiled `fastfields-dlpack` build, and `cpu` is the right one
here. The same `fastfields` distribution also ships `fastfields.torch`,
`fastfields.cupy`, `fastfields.auto` and `fastfields.helpers`.

## A first example

```python
import numpy as np
import fastfields.numpy as ff

feature = np.full((256, 256), np.inf, dtype="float32")
feature[:, 128] = 0.0  # features are 0, everything else is +inf

dist = ff.dt_euclidean(feature)  # squared distance to column 128, per row
```

Distance transforms work along the last axis and expect `0` at the features
and `+inf` elsewhere. `dt_euclidean` returns squared distances.

## What's inside

| Operation | Functions |
|---|---|
| **Distance transforms** | [`dt_euclidean`][fastfields.numpy.dt_euclidean] and [`dt_l1`][fastfields.numpy.dt_l1] along the last axis; point-to-spline [`dt_spline_table`][fastfields.numpy.dt_spline_table], [`dt_spline_brent`][fastfields.numpy.dt_spline_brent] and [`dt_spline_gaussnewton`][fastfields.numpy.dt_spline_gaussnewton]; point-to-mesh [`dt_mesh`][fastfields.numpy.dt_mesh] |
| **Positive-definite linear algebra** | [`sym_matvec`][fastfields.numpy.sym_matvec], [`sym_addmatvec_`][fastfields.numpy.sym_addmatvec_], [`sym_submatvec_`][fastfields.numpy.sym_submatvec_], [`sym_solve`][fastfields.numpy.sym_solve] and [`sym_invert`][fastfields.numpy.sym_invert], over whole fields of small symmetric matrices |
| **Resampling** | [`resample`][fastfields.numpy.resample] (spline up- and down-sampling), [`restriction`][fastfields.numpy.restriction] (its adjoint) and [`spline_coeff`][fastfields.numpy.spline_coeff] (the coefficient prefilter) |
| **Spline sampling (pushpull)** | [`pull`][fastfields.numpy.pull] (gather), [`push`][fastfields.numpy.push] (scatter), [`count`][fastfields.numpy.count] and [`grad`][fastfields.numpy.grad] |
| **Regularisers** | `field_*` for a multi-channel field, with one penalty per channel, and `flow_*` for a vector flow, with scalar penalties plus the Lamé `shears` and `div` terms. Each family has the operator (`*_matvec`), its diagonal (`*_diag`), its stencil (`*_kernel`), accumulate forms of each (`*_add…` and `*_sub…`), weighted variants (`*_rls`) and the regularised system described below |

Anything that selects a spline order or a boundary condition (`order`,
`bound`) accepts an `int`, a [`Spline`][fastfields.numpy.Spline] or
[`Bound`][fastfields.numpy.Bound] member, or a name such as `"cubic"` or
`"dct2"`.

A *field* is a multi-channel array with any number of channels. A *flow* is a
vector field whose number of channels equals the number of spatial dimensions.
The regularisers act on the `ndim` axes just before the channel axis, and
`ndim` is not inferred: it defaults to 1, so pass `ndim=2` for a 2D image.

### Solving the regularised system

Let `H` be the per-voxel Hessian of a data term, stored as compact-symmetric
matrices, and `L` a regulariser. These functions apply and solve the system
`(H + L) x = g`:

| call | what it does |
|---|---|
| [`field_forward`][fastfields.numpy.field_forward], [`flow_forward`][fastfields.numpy.flow_forward] | apply `(H + L) x` |
| [`field_precond`][fastfields.numpy.field_precond], [`flow_precond`][fastfields.numpy.flow_precond] | apply `(H + diag(L))⁻¹ g`, the Jacobi preconditioner |
| [`field_precond_`][fastfields.numpy.field_precond_], [`flow_precond_`][fastfields.numpy.flow_precond_] | the same, in place |
| [`field_relax`][fastfields.numpy.field_relax], [`flow_relax`][fastfields.numpy.flow_relax] | Gauss-Seidel sweeps on the system |
| [`field_cg`][fastfields.numpy.field_cg] | preconditioned conjugate gradients on the same system |

`field_cg` updates `sol` in place and returns it, like `field_relax`. Pass
`return_info=True` to also get the iteration count and the final relative
residual. The solver is CPU-only in the underlying library, which is why
`fastfields.cupy` has no `field_cg`.

## Batch axes and memory

Leading axes are batch axes, and they broadcast: you can combine a single
matrix field with a stack of vector fields, and the result carries the
broadcast batch shape. The broadcast is a view with zero strides, so nothing is
copied. Inputs are likewise passed with their own strides instead of being made
contiguous. Only read-only arrays are copied, because NumPy cannot export them
through DLPack.

## What this package does not wrap

Of the 58 operations of the underlying C++ library, this package wraps all but
four: `pull_backward`, `push_backward`, `count_backward` and `grad_backward`.
They are the adjoints of the sampling operators with respect to both the
sampled field and the sampling coordinates, and they exist so that an autograd
engine can differentiate through `grid`. Only `fastfields.torch` has one, so
NumPy has no use for them. They stay at the raw layer, as
`fastfields.dlpack.pull_backward(...)` and so on.
`tests/test_api_coverage_numpy.py` fails if that list ever changes by accident.

See the [API reference](api/numpy.md) for the full signatures.
