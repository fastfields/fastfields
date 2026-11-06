# fastfields.torch

`fastfields.torch` runs the fastfields operators on PyTorch tensors, and most
of them are differentiable: put them inside a model and gradients flow
straight through. Functions take tensors and return newly allocated tensors.

## Install

```sh
pip install "fastfields[torch]" \
    --extra-index-url https://fastfields.github.io/whl/cpu/
```

CPU tensors work with any build of `fastfields-dlpack`. A CUDA tensor needs a
CUDA build, which means installing from one of the `cu*` lanes of the index
instead of `cpu/`; then the same code runs on the GPU. Handing a CUDA tensor to
a CPU-only build raises a `RuntimeError` that names the lane to install. The
check happens when the tensor arrives and not at import, so the CPU path keeps
working everywhere.

## A first example

```python
import torch
import fastfields.torch as ff

mat = torch.randn(5, 6, dtype=torch.float64, requires_grad=True)  # packed, C=3
vec = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)

out = ff.sym_matvec(mat, vec)  # H @ vec, one 3x3 matrix per row
out.sum().backward()  # gradients land on both mat and vec
```

## What's inside

| Operation | Functions |
|---|---|
| **Positive-definite linear algebra** | [`sym_matvec`][fastfields.torch.sym_matvec] and [`sym_solve`][fastfields.torch.sym_solve] (differentiable); [`sym_invert`][fastfields.torch.sym_invert], [`sym_addmatvec_`][fastfields.torch.sym_addmatvec_] and [`sym_submatvec_`][fastfields.torch.sym_submatvec_], over whole fields of small symmetric matrices |
| **Resampling** | [`resample`][fastfields.torch.resample] (spline up- and down-sampling), [`restriction`][fastfields.torch.restriction] (its adjoint) and [`spline_coeff`][fastfields.torch.spline_coeff] (the coefficient prefilter), all differentiable |
| **Spline sampling (pushpull)** | [`pull`][fastfields.torch.pull], [`push`][fastfields.torch.push], [`count`][fastfields.torch.count] and [`grad`][fastfields.torch.grad], all differentiable with respect to the sampling `grid` |
| **Distance transforms** | [`dt_euclidean`][fastfields.torch.dt_euclidean], [`dt_l1`][fastfields.torch.dt_l1], [`dt_mesh`][fastfields.torch.dt_mesh] and point-to-spline [`dt_spline_table`][fastfields.torch.dt_spline_table], [`dt_spline_brent`][fastfields.torch.dt_spline_brent] and [`dt_spline_gaussnewton`][fastfields.torch.dt_spline_gaussnewton] |
| **Regularisers** | `field_*` for a multi-channel field, with one penalty per channel, and `flow_*` for a vector flow, with scalar penalties plus the Lamé `shears` and `div` terms. Each family has the operator (`*_matvec`), its diagonal (`*_diag`), its stencil (`*_kernel`), accumulate forms of each (`*_add…` and `*_sub…`), weighted variants (`*_rls`), `*_relax`, and `*_forward` and `*_precond` |

A *field* is a multi-channel tensor with any number of channels. A *flow* is a
vector field whose number of channels equals the number of spatial dimensions.
The regularisers act on the `ndim` axes just before the channel axis, and
`ndim` is not inferred: it defaults to 1, so pass `ndim=2` for a 2D image.

## What is differentiable

These operations support `.backward()`:

- `sym_matvec`, with respect to `mat` and `vec`, and `sym_solve`, with respect
  to `vec` only (`mat` must not require grad);
- `resample`, `restriction` and `spline_coeff`, with respect to the input;
- all four sampling operators, with respect to the field and the sampling
  coordinates (`count` only has the coordinates);
- the regulariser operators, including `field_forward` and `field_precond` and
  their flow twins.

`grad` deserves a remark. It is itself a derivative operator, so
differentiating it produces second-order terms. That is the point: an image
gradient computed inside a loss has to be differentiable for the deformation
that produced `grid` to be learnable. Its `abs=True` mode is a majorisation,
and its backward pass is the adjoint of the majorised operator, not a true
Jacobian. Use it as the majorisation it is.

The remaining operations are exposed for parity with `fastfields.numpy` and
`fastfields.cupy` but are not differentiable: the distance transforms
(`dt_euclidean`, `dt_l1`, `dt_mesh` and the three `dt_spline_*`), `sym_invert`,
the iterative solvers `field_relax` and `flow_relax` with their weighted
variants, and the gradients of `sym_addmatvec_` and `sym_submatvec_` with
respect to `mat` and `vec`. Their forward pass always runs, even when an input
requires grad. Only calling `.backward()` through the output raises a
`RuntimeError` that names the operation.

## What this package does not wrap

Of the 58 operations of the underlying C++ library, this package wraps all but
eight, in two groups.

**Adjoints that are used, not re-exported.** `pull_backward`, `push_backward`,
`count_backward`, `grad_backward` and `sym_matvec_backward` are the gradients
behind the `autograd.Function` of each forward operation. The supported way to
reach them is to call the forward operation and then `.backward()`.
Re-exporting them would be a second, unchecked spelling of a gradient, with
pre-zeroed buffers and an accumulate-or-overwrite contract for the caller to
get right. `fastfields.dlpack` has them raw.

**Three solver entry points that need an autograd decision first.**
`field_precond_` and `flow_precond_` are in-place preconditioner solves, and
`field_cg` runs an iterative solver whose gradient is a solve with the same
operator. Both are worth doing and neither is a mechanical wrapper, so they are
not here yet. `field_relax` is the in-place solver available today, and
`fastfields.numpy` has all three for the case without autograd.

`tests/test_api_coverage_torch.py` fails if that list ever changes by
accident.

See the [API reference](api/torch.md) for the full signatures.
