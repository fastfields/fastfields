"""Spline resampling, restriction and coefficient prefilter, with autograd.

Wraps ``fastfields.dlpack.resample`` / ``restriction`` (prolongation and its
adjoint) and ``fastfields.dlpack.spline_coeff`` (interpolating-coefficient
prefilter along the last axis).

Adjoint relationship (verified empirically against the bindings, and matching
the spirit of ``jitfields`` ``resize.py`` where the backward of ``resize`` is
``restrict`` and vice-versa)::

    transpose( resample(scale=s, shift=h) ) == restriction(scale=1/s, shift=h)

i.e. the adjoint uses the **reciprocal** scale and the **same** shift, with the
input/output shapes swapped. The backward passes below implement exactly that.
The spline-coefficient prefilter is linear, so its backward applies the
transpose of the prefilter to the gradient. That transpose is the prefilter
itself for ``replicate``/``dct2``/``dft``, but **not** for ``dct1``, whose
operator is only self-adjoint under the endpoint-weighted inner product
``W = diag(1/2, 1, ..., 1, 1/2)`` -- see ``_spline_coeff_adjoint``.

!!! note
    * ``scale`` uses the binding convention "input-index per output-index".
    * This wrapper always passes the binding an explicit per-dim ``scale`` --
      derived from ``anchor`` and the shapes, or from an explicit ``scale``
      override (see [`resample`][fastfields.torch.resample]). (The binding
      itself now handles ``scale=None`` by defaulting to ``inshape /
      outshape``; the wrapper resolves the scale up front so the ``anchor``
      conventions are applied consistently.)
    * On CUDA tensors the current stream is forwarded to the binding (see
      `fastfields.torch._util`).
"""

from __future__ import annotations

from typing import Optional, Sequence

import fastfields.dlpack as _fb

import torch
from fastfields.helpers import (
    Bound,
    anchor_scale_shift,
    as_bound,
    as_spline,
    check_ndim,
    infer_ndim,
    resolve_out_spatial,
)
from torch import Tensor

from ._util import check_dtype, check_inplace_leaf, stream_ptr

__all__ = ["resample", "restriction", "spline_coeff", "spline_coeff_"]


def _effective_scale(
    scale: Optional[Sequence[float]],
    inshape: Sequence[int],
    outshape: Sequence[int],
    ndim: int,
) -> list[float]:
    """Resolve the per-dim scale, defaulting to ``inshape/outshape``.

    Parameters
    ----------
    scale : sequence of float, optional
        Explicit per-dim scale (input-index per output-index). When ``None``,
        it is computed from the shapes.
    inshape, outshape : sequence of int
        Input and output spatial shapes.
    ndim : int
        Number of spatial dimensions.

    Returns
    -------
    list of float
        The per-dim scale, length ``ndim``.

    Raises
    ------
    ValueError
        If an explicit ``scale`` does not have length ``ndim``.
    """
    if scale is None:
        return [float(inshape[d]) / float(outshape[d]) for d in range(ndim)]
    scale = list(scale)
    if len(scale) != ndim:
        raise ValueError(f"Expected scale of length ndim={ndim}, got {scale}.")
    return [float(s) for s in scale]


def resample(
    inp: Tensor,
    factor: float | Sequence[float] | None = None,
    shape: int | Sequence[int] | None = None,
    *,
    order: int | str = 2,
    bound: int | str = "dct2",
    ndim: Optional[int] = None,
    anchor: str = "centers",
    shift: Optional[float] = None,
    scale: Optional[Sequence[float]] = None,
) -> Tensor:
    """Resample the trailing axes of an array with splines.

    Resamples `inp` onto a new grid with splines of order `order`, along the
    last `ndim` axes; leading axes are batch axes. Give either a size `factor`
    or an explicit output `shape`, not both.

    Which points of the two grids line up is set by `anchor`; with `"centers"`,
    the default, the first and last samples coincide. The `anchor` determines
    the sampling `scale` and `shift` for the kernel, and either can be
    overridden.

    [`restriction`][fastfields.torch.restriction] is the adjoint operation.

    Differentiable with respect to `inp`; the backward pass is
    [`restriction`][fastfields.torch.restriction].

    !!! tip "Interpolating resize"
        `resample` does not prefilter its input. For a true interpolating
        resize, pass the coefficients computed by
        [`spline_coeff`][fastfields.torch.spline_coeff] instead of the raw
        samples.

    Parameters
    ----------
    inp : torch.Tensor
        Input, shape `(*batch, *spatial_in)`; the last `ndim` axes are
        resampled.
    factor : float or sequence of float, optional
        Size multiplier, one per resampled axis or a single value for all.
        Exclusive with `shape`; with neither, the spatial shape is
        unchanged. Each output size is `max(1, round(n * factor))`.
    shape : int or sequence of int, optional
        Output spatial size, one per resampled axis or a single value for
        all. Exclusive with `factor`.
    order : int or str, default=2
        Spline order: an integer from 0 to 7, a
        [`Spline`][fastfields.helpers.Spline] member, or a name such as
        `"cubic"`.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, optional
        Number of trailing axes to resample. When omitted it is the length
        of `shape` or `factor` if either is a sequence, and 1 otherwise.
    anchor : {"centers", "edges", "first", "last"}, default="centers"
        Which points of the input and output grids are aligned; the initial
        letter is enough. See
        [`anchor_scale_shift`][fastfields.helpers.anchor_scale_shift] for
        the exact `scale` and `shift` each choice implies.
    shift : float, optional
        Sampling shift, replacing the one implied by `anchor`.
    scale : sequence of float, optional
        Input-index step per output-index step, one value per resampled
        axis, replacing the one derived from `anchor` and the shapes.

    Returns
    -------
    torch.Tensor
        The resampled tensor, shape `(*batch, *spatial_out)`.

    Raises
    ------
    ValueError
        If `ndim` is outside `1..inp.dim()`, `anchor`, `order` or `bound`
        is unknown, or `factor`, `shape` or `scale` has the wrong length.
    """
    check_dtype(inp)
    ndim = infer_ndim(ndim, factor, shape)
    check_ndim(ndim, inp.dim())
    spatial_in = tuple(inp.shape[-ndim:])
    out_spatial = resolve_out_spatial(spatial_in, ndim, factor, shape)
    a_scale, a_shift = anchor_scale_shift(
        anchor, spatial_in, out_spatial, ndim
    )
    if scale is not None:
        a_scale = _effective_scale(scale, spatial_in, out_spatial, ndim)
    if shift is not None:
        a_shift = float(shift)
    return _Resample.apply(
        inp,
        list(out_spatial),
        as_spline(order),
        as_bound(bound),
        a_shift,
        a_scale,
        ndim,
    )


def restriction(
    inp: Tensor,
    factor: float | Sequence[float] | None = None,
    shape: int | Sequence[int] | None = None,
    *,
    order: int | str = 2,
    bound: int | str = "dct2",
    ndim: Optional[int] = None,
    anchor: str = "centers",
    shift: Optional[float] = None,
    scale: Optional[Sequence[float]] = None,
) -> Tensor:
    """Adjoint of `resample`, along the trailing axes of an array.

    Transposes the linear map of [`resample`][fastfields.torch.resample] for
    the same arguments: values on the finer grid are accumulated onto the
    coarser one. The `scale` is derived from this call's own input and output
    shapes, so a `resample` and a `restriction` with matching shapes use
    reciprocal scales and the same shift.

    The binding accumulates into its output buffer, which is why the buffer is
    allocated zero-filled here. When calling `fastfields.dlpack.restriction`
    yourself, pre-zero `out`.

    `factor`, `shape`, `order`, `bound`, `ndim`, `anchor`, `shift` and `scale`
    are exactly as for [`resample`][fastfields.torch.resample].

    Differentiable with respect to `inp`; the backward pass is
    [`resample`][fastfields.torch.resample].

    Parameters
    ----------
    inp : torch.Tensor
        Input, shape `(*batch, *spatial_in)`; the last `ndim` axes are
        resampled.
    factor : float or sequence of float, optional
        Size multiplier, as for [`resample`][fastfields.torch.resample].
        Exclusive with `shape`.
    shape : int or sequence of int, optional
        Output spatial size, as for
        [`resample`][fastfields.torch.resample]. Exclusive with `factor`.
    order : int or str, default=2
        Spline order: an integer from 0 to 7, a
        [`Spline`][fastfields.helpers.Spline] member, or a name such as
        `"cubic"`.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, optional
        Number of trailing axes, as for
        [`resample`][fastfields.torch.resample].
    anchor : {"centers", "edges", "first", "last"}, default="centers"
        Grid alignment, as for [`resample`][fastfields.torch.resample].
    shift : float, optional
        Sampling shift, as for [`resample`][fastfields.torch.resample].
    scale : sequence of float, optional
        Per-axis scale, as for [`resample`][fastfields.torch.resample].

    Returns
    -------
    torch.Tensor
        The restricted tensor, shape `(*batch, *spatial_out)`.

    Raises
    ------
    ValueError
        If `ndim` is outside `1..inp.dim()`, `anchor`, `order` or `bound`
        is unknown, or `factor`, `shape` or `scale` has the wrong length.
    """
    check_dtype(inp)
    ndim = infer_ndim(ndim, factor, shape)
    check_ndim(ndim, inp.dim())
    spatial_in = tuple(inp.shape[-ndim:])
    out_spatial = resolve_out_spatial(spatial_in, ndim, factor, shape)
    a_scale, a_shift = anchor_scale_shift(
        anchor, spatial_in, out_spatial, ndim
    )
    if scale is not None:
        a_scale = _effective_scale(scale, spatial_in, out_spatial, ndim)
    if shift is not None:
        a_shift = float(shift)
    return _Restriction.apply(
        inp,
        list(out_spatial),
        as_spline(order),
        as_bound(bound),
        a_shift,
        a_scale,
        ndim,
    )


def spline_coeff(
    inp: Tensor, order: int | str = 3, bound: int | str = "dct2"
) -> Tensor:
    """Spline coefficients of a signal, along the last axis.

    Converts samples into the coefficients of the interpolating spline of order
    `order`, so that evaluating that spline at the sample positions gives the
    samples back. For orders 0 and 1 the coefficients are the samples. Only the
    last axis is filtered.

    For orders above 1 only the boundary conditions `dct1`, `dct2`, `dft` and
    `replicate` are implemented; any other is rejected. The coefficients are
    what [`pull`][fastfields.torch.pull], [`push`][fastfields.torch.push] and
    [`grad`][fastfields.torch.grad] expect as their volume.

    [`spline_coeff_`][fastfields.torch.spline_coeff_] is the in-place version.

    Differentiable with respect to `inp`. An unimplemented boundary condition
    raises `ValueError`.

    Parameters
    ----------
    inp : torch.Tensor
        Samples; the transform runs along the last axis.
    order : int or str, default=3
        Spline order: an integer from 0 to 7, a
        [`Spline`][fastfields.helpers.Spline] member, or a name such as
        `"cubic"`.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.

    Returns
    -------
    torch.Tensor
        The coefficients, a new tensor the shape of `inp`.

    Raises
    ------
    ValueError
        If `bound` is not an implemented boundary condition.
    """
    check_dtype(inp)
    spline, bound = as_spline(order), as_bound(bound)
    _check_splinc_bound(spline, bound)
    return _SplineCoeff.apply(inp, spline, bound)


def spline_coeff_(
    inp_out: Tensor, order: int | str = 3, bound: int | str = "dct2"
) -> Tensor:
    """Spline coefficients of a signal along the last axis, in place.

    The in-place version of [`spline_coeff`][fastfields.torch.spline_coeff]:
    the samples in `inp_out` are replaced by the coefficients. For orders above
    1 only the boundary conditions `dct1`, `dct2`, `dft` and `replicate` are
    implemented; any other is rejected.

    Differentiable with respect to `inp_out`: the prefilter is linear, and its
    backward pass never reads the overwritten values. An unimplemented boundary
    condition raises `ValueError`. A leaf tensor that requires grad cannot be
    modified in place; this is checked before `inp_out` is touched.

    Parameters
    ----------
    inp_out : torch.Tensor
        Samples on entry, spline coefficients on exit. Updated in place.
    order : int or str, default=3
        Spline order: an integer from 0 to 7, a
        [`Spline`][fastfields.helpers.Spline] member, or a name such as
        `"cubic"`.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.

    Returns
    -------
    torch.Tensor
        `inp_out` itself, now holding the coefficients.

    Raises
    ------
    ValueError
        If `bound` is not an implemented boundary condition.
    RuntimeError
        If `inp_out` is a leaf tensor that requires grad.
    """
    check_dtype(inp_out)
    check_inplace_leaf("spline_coeff_", inp_out)
    spline, bound = as_spline(order), as_bound(bound)
    _check_splinc_bound(spline, bound)
    return _SplineCoeffInPlace.apply(inp_out, spline, bound)


# --------------------------------------------------------------------------- #
# spline_coeff: supported bounds, and the adjoint used by backward
# --------------------------------------------------------------------------- #
#
# The prefilter is linear, so its backward is the transpose of the operator
# ``M`` it applies. Whether ``M`` is *symmetric* -- i.e. whether the backward
# may simply re-apply the same prefilter -- depends on the boundary condition,
# and it is NOT symmetric for every bound the binding accepts:
#
#   bound       max|M - M^T|   backward = prefilter(grad)?
#   replicate   ~1e-16         yes (symmetric)
#   dct2        ~1e-16         yes (symmetric)
#   dft         ~1e-16         yes (symmetric)
#   dct1        O(1)           NO -- needs the weighted adjoint below
#
# For DCT-I (whole-point symmetry) the endpoints are shared by the mirrored
# extension, so ``M`` is self-adjoint with respect to the weighted inner
# product ``W = diag(1/2, 1, ..., 1, 1/2)`` rather than the plain Euclidean
# one: ``W M == (W M)^T``, hence ``M^T == W M W^-1``. Re-applying the plain
# prefilter there returns a *silently wrong* gradient (verified: the error is
# O(1) and does not shrink with the axis length).
_BOUND_DCT1 = int(Bound.DCT1)

# Bounds the prefilter is actually implemented for (mirrors jitfields'
# ``splinc.checkbound``). Everything else raises inside the binding itself
# now (fastfields-lib#70 -- previously ``zero`` was silently aliased to
# ``dct1``, handing back results for a boundary condition the caller did
# not ask for). This check is kept as defense-in-depth: it still applies
# to the built-in ``dst1``/``dst2`` bounds, and it gives a
# clear error to callers on older published wheels that predate #70,
# where the binding still aliases ``zero`` silently instead of raising.
_SPLINC_BOUNDS_OK = (
    int(Bound.DCT1),
    int(Bound.DCT2),
    int(Bound.DFT),
    int(Bound.Replicate),
)


def _check_splinc_bound(spline: int, bound: int) -> None:
    """Reject boundary conditions the prefilter does not implement.

    Orders 0 and 1 make the prefilter the identity, so the bound is
    irrelevant there (same carve-out as jitfields).

    Raises
    ------
    ValueError
        If ``bound`` is not one of the implemented boundary conditions.
    """
    if spline <= 1 or bound in _SPLINC_BOUNDS_OK:
        return
    ok = ", ".join(Bound(b).name.lower() for b in _SPLINC_BOUNDS_OK)
    raise ValueError(
        f"spline_coeff is only implemented for bounds ({ok}), got "
        f"{Bound(bound).name.lower()!r}."
    )


def _spline_coeff_adjoint(grad: Tensor, spline: int, bound: int) -> Tensor:
    """Apply the transpose of the prefilter to ``grad`` (a new tensor).

    Parameters
    ----------
    grad : torch.Tensor
        Incoming gradient, shape ``(..., N)``.
    spline : int
        Spline order (already resolved to an int).
    bound : int
        Boundary condition (already resolved to an int).

    Returns
    -------
    torch.Tensor
        ``M^T @ grad``, where ``M`` is the prefilter applied by forward.
    """
    gout = grad.clone()
    if bound == _BOUND_DCT1:
        # M^T = W M W^-1 with W = diag(1/2, 1, ..., 1, 1/2). Orders <= 1 make
        # the prefilter the identity, and the two scalings then cancel, so
        # this stays correct there too.
        gout[..., 0] *= 2.0
        gout[..., -1] *= 2.0
        _fb.spline_coeff(gout, spline, bound, stream=stream_ptr(gout))
        gout[..., 0] *= 0.5
        gout[..., -1] *= 0.5
    else:
        _fb.spline_coeff(gout, spline, bound, stream=stream_ptr(gout))
    return gout


def _do_resample(
    out: Tensor,
    inp: Tensor,
    spline: int,
    bound: int,
    shift: float,
    scale: Sequence[float],
    ndim: int,
) -> None:
    """Call the resample binding, forwarding ``out``'s CUDA stream."""
    _fb.resample(
        out, inp, spline, bound, shift, scale, ndim, stream=stream_ptr(out)
    )


def _do_restriction(
    out: Tensor,
    inp: Tensor,
    spline: int,
    bound: int,
    shift: float,
    scale: Sequence[float],
    ndim: int,
) -> None:
    """Call the restriction binding (accumulating), forwarding the stream."""
    # `restriction` accumulates into `out`; it must be pre-zeroed.
    out.zero_()
    _fb.restriction(
        out, inp, spline, bound, shift, scale, ndim, stream=stream_ptr(out)
    )


def _reciprocal(scale: Sequence[float]) -> list[float]:
    """Return the element-wise reciprocal of ``scale``."""
    return [1.0 / s for s in scale]


class _Resample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, shape, spline, bound, shift, scale, ndim):
        # inp is a read-only input: the stride-aware binding reads it
        # zero-copy, so we do NOT force contiguity. The output buffer
        # (new_empty) is contiguous regardless of inp's layout.
        out = inp.new_empty([*inp.shape[:-ndim], *shape])
        _do_resample(out, inp, spline, bound, shift, scale, ndim)
        ctx.opt = (list(inp.shape[-ndim:]), spline, bound, shift, scale, ndim)
        return out

    @staticmethod
    def backward(ctx, grad):
        inshape, spline, bound, shift, scale, ndim = ctx.opt
        ginp = None
        if ctx.needs_input_grad[0]:
            ginp = grad.new_empty([*grad.shape[:-ndim], *inshape])
            # adjoint of resample(s) is restriction(1/s, same shift)
            _do_restriction(
                ginp, grad, spline, bound, shift, _reciprocal(scale), ndim
            )
        return (ginp,) + (None,) * 6


class _Restriction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, shape, spline, bound, shift, scale, ndim):
        # See _Resample.forward: read-only input, contiguous output.
        out = inp.new_empty([*inp.shape[:-ndim], *shape])
        _do_restriction(out, inp, spline, bound, shift, scale, ndim)
        ctx.opt = (list(inp.shape[-ndim:]), spline, bound, shift, scale, ndim)
        return out

    @staticmethod
    def backward(ctx, grad):
        inshape, spline, bound, shift, scale, ndim = ctx.opt
        ginp = None
        if ctx.needs_input_grad[0]:
            ginp = grad.new_empty([*grad.shape[:-ndim], *inshape])
            # adjoint of restriction(s) is resample(1/s, same shift)
            _do_resample(
                ginp, grad, spline, bound, shift, _reciprocal(scale), ndim
            )
        return (ginp,) + (None,) * 6


class _SplineCoeff(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, spline, bound):
        # Layout-preserving copy: the binding filters in place through the
        # tensor's strides, so this clone is the output buffer (its layout
        # matches inp -- the stride-aware binding needs no contiguous copy).
        out = inp.clone()
        _fb.spline_coeff(out, spline, bound, stream=stream_ptr(out))
        ctx.spline = spline
        ctx.bound = bound
        return out

    @staticmethod
    def backward(ctx, grad):
        gout = None
        if ctx.needs_input_grad[0]:
            gout = _spline_coeff_adjoint(grad, ctx.spline, ctx.bound)
        return gout, None, None


class _SplineCoeffInPlace(torch.autograd.Function):
    """``inp <- spline_coeff(inp)`` in place.

    Autograd-safe for the same reason as `_SplineCoeff`: the
    prefilter is linear and its backward only needs the saved
    ``spline``/``bound`` scalars, never the pre-mutation ``inp`` --
    ``ctx.mark_dirty(inp)`` bumps the version counter so a stale save
    elsewhere raises instead of silently returning a wrong gradient.
    """

    @staticmethod
    def forward(ctx, inp, spline, bound):
        _fb.spline_coeff(inp, spline, bound, stream=stream_ptr(inp))
        ctx.mark_dirty(inp)
        ctx.spline = spline
        ctx.bound = bound
        return inp

    @staticmethod
    def backward(ctx, grad):
        gout = None
        if ctx.needs_input_grad[0]:
            gout = _spline_coeff_adjoint(grad, ctx.spline, ctx.bound)
        return gout, None, None
