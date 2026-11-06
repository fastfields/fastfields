# fastfields.cupy

`fastfields.cupy` runs the fastfields operators on the GPU, directly on CuPy
arrays. The API is the one of `fastfields.numpy`, but the work happens in
device memory and is ordered on CuPy's current CUDA stream.

CuPy itself is not installed automatically, because the right build depends on
your CUDA version. Install it with the `cupy` extra, or install a matching
build such as `cupy-cuda12x` yourself.

## Install

```sh
pip install "fastfields[cupy]" \
    --extra-index-url https://fastfields.github.io/whl/cu130/
```

!!! warning "The extra selects CuPy; the index URL selects the CUDA kernels"
    These are two separate choices, and only the second one can fail quietly.
    `pip install "fastfields[cupy]"` installs CuPy and succeeds even against a
    CPU-only `fastfields-dlpack`, because `pip` has no way to say "any build
    with CUDA": the lane comes from the index folder you point it at.
    `import fastfields.cupy` then fails at once, with a message naming the lane
    to install, because there is no CPU path to fall back to. See [the
    packaging note](monorepo.md) for why it works this way.

    No `cu*` lane is published yet. Until one is, this install has nothing to
    resolve to. The [wheel index](https://fastfields.github.io/whl/) lists the
    lanes that exist.

## A first example

```python
import cupy as cp
import fastfields.cupy as ff

feature = cp.full((256, 256), cp.inf, dtype="float32")
feature[:, 128] = 0.0  # features are 0, everything else is +inf

dist = ff.dt_euclidean(feature)  # a new array; feature is untouched
ff.dt_euclidean_(feature)  # the in-place variant writes into feature
```

## What's inside

| Operation | Functions |
|---|---|
| **Distance transforms** | [`dt_euclidean`][fastfields.cupy.dt_euclidean] and [`dt_l1`][fastfields.cupy.dt_l1] along the last axis; point-to-spline [`dt_spline_table`][fastfields.cupy.dt_spline_table], [`dt_spline_brent`][fastfields.cupy.dt_spline_brent] and [`dt_spline_gaussnewton`][fastfields.cupy.dt_spline_gaussnewton]; point-to-mesh [`dt_mesh`][fastfields.cupy.dt_mesh] |
| **Positive-definite linear algebra** | [`sym_matvec`][fastfields.cupy.sym_matvec], [`sym_addmatvec_`][fastfields.cupy.sym_addmatvec_], [`sym_submatvec_`][fastfields.cupy.sym_submatvec_], [`sym_solve`][fastfields.cupy.sym_solve] and [`sym_invert`][fastfields.cupy.sym_invert], over whole fields of small symmetric matrices |
| **Resampling** | [`resample`][fastfields.cupy.resample] (spline up- and down-sampling), [`restriction`][fastfields.cupy.restriction] (its adjoint) and [`spline_coeff`][fastfields.cupy.spline_coeff] (the coefficient prefilter) |
| **Spline sampling (pushpull)** | [`pull`][fastfields.cupy.pull] (gather), [`push`][fastfields.cupy.push] (scatter), [`count`][fastfields.cupy.count] and [`grad`][fastfields.cupy.grad] |
| **Regularisers** | `field_*` for a multi-channel field, with one penalty per channel, and `flow_*` for a vector flow, with scalar penalties plus the Lamé `shears` and `div` terms. Each family has the operator (`*_matvec`), its diagonal (`*_diag`), its stencil (`*_kernel`), accumulate forms of each (`*_add…` and `*_sub…`), weighted variants (`*_rls`) and the regularised system described below |

The functions without a trailing underscore allocate their outputs and return
new arrays. The ones with a trailing underscore (`dt_euclidean_`, `sym_solve_`
and so on) write in place. To run on a specific stream, call the function
inside a `with my_stream:` block.

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
| [`field_forward`][fastfields.cupy.field_forward], [`flow_forward`][fastfields.cupy.flow_forward] | apply `(H + L) x` |
| [`field_precond`][fastfields.cupy.field_precond], [`flow_precond`][fastfields.cupy.flow_precond] | apply `(H + diag(L))⁻¹ g`, the Jacobi preconditioner |
| [`field_precond_`][fastfields.cupy.field_precond_], [`flow_precond_`][fastfields.cupy.flow_precond_] | the same, in place |
| [`field_relax`][fastfields.cupy.field_relax], [`flow_relax`][fastfields.cupy.flow_relax] | Gauss-Seidel sweeps on the system |

Each of these is a single fused kernel, so none of them allocates a temporary
that you would then have to keep alive until the stream is synchronised.

## What this package does not wrap

Of the 58 operations of the underlying C++ library, this package wraps all but
five.

`pull_backward`, `push_backward`, `count_backward` and `grad_backward` are the
adjoints of the sampling operators with respect to both the sampled field and
the sampling coordinates. They exist so that an autograd engine can
differentiate through `grid`, which only `fastfields.torch` does. CuPy has no
automatic differentiation, so they stay at the raw layer, as
`fastfields.dlpack.pull_backward(...)` and so on.

`field_cg`, the conjugate-gradient solver, is CPU-only in the underlying
library. Its dot products need a device-side reduction that the CUDA backend
does not implement yet ([fastfields-lib#34][lib34]), and the hub rejects CUDA
arrays instead of quietly running on the host. A `fastfields.cupy.field_cg`
could therefore only ever raise, so there isn't one. Use `field_relax`, which
runs on CUDA and solves the same system by Gauss-Seidel sweeps.

`tests/test_api_coverage_cupy.py` fails if that list ever changes by accident,
and it runs without a GPU.

See the [API reference](api/cupy.md) for the full signatures.

[lib34]: https://github.com/fastfields/fastfields-lib/issues/34
