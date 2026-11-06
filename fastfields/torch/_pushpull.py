"""Spline-interpolation gather / scatter (pushpull) — torch.

Channel-last, x-first coordinate convention (spatial rank ``D`` = ``grid``'s
trailing axis; ``B`` leading batch dims):

* ``inp``  : ``(*B, *inshape,  C)``   — the volume (spline coefficients)
* ``grid`` : ``(*B, *outshape, D)``   — sampling coordinates, in voxels

``pull`` and ``push`` are differentiable with respect to **both** the field and
the ``grid``: differentiating through the sample positions is what makes them
usable inside a learned deformation / registration model.

``count`` is differentiable with respect to ``grid`` (its only input) and
``grad`` with respect to both ``inp`` and ``grid`` -- via the hub's
``count_backward`` / ``grad_backward`` adjoints, which this module had been
leaving on the table. ``grad`` is already a derivative operator, so
differentiating *it* gives second-order terms; that is exactly what a
registration model computing an image gradient inside its loss needs, and
``grad_backward`` is the kernel for it.

Every one of the four operators is therefore differentiable here, and all
four pushpull adjoints in ``fastfields.dlpack`` now have a user. The ``abs``
flag of ``grad`` is the one caveat: it replaces the signed spline derivative
by its absolute value (a majorisation trick), and ``grad_backward`` must be
called with the same flag -- these wrappers do that for you.
"""

from __future__ import annotations

from typing import Sequence

import fastfields.dlpack as _fb

import torch
from fastfields.helpers import as_bound, as_spline
from torch import Tensor

from ._util import check_dtype, stream_ptr

__all__ = ["pull", "push", "count", "grad"]


def _spatial(shape: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,) * ndim
    out = tuple(int(s) for s in shape)
    if len(out) != ndim:
        raise ValueError(f"shape must have length ndim={ndim}, got {shape!r}")
    return out


class _Pull(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, grid, spline, bound, extrapolate):
        out = inp.new_zeros((*grid.shape[:-1], inp.shape[-1]))
        _fb.pull(
            out,
            inp,
            grid,
            spline=spline,
            bound=bound,
            extrapolate=extrapolate,
            stream=stream_ptr(out),
        )
        d = grid.shape[-1]
        nbatch = grid.ndim - d - 1
        # `inp` is only needed to differentiate wrt the sample positions
        # (d(pull)/d(grid) is the spatial gradient of the field); skip saving
        # it otherwise so the field-only path keeps its old memory profile.
        ctx.save_for_backward(inp if grid.requires_grad else None, grid)
        ctx.spline, ctx.bound, ctx.extrapolate = spline, bound, extrapolate
        ctx.inshape = tuple(inp.shape[nbatch : nbatch + d])
        ctx.channels = inp.shape[-1]
        return out

    # The backward calls straight into the C++ adjoints, which are opaque to
    # autograd -- so it is not itself differentiable. Say so explicitly rather
    # than silently returning a wrong second derivative under create_graph.
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        inp, grid = ctx.saved_tensors
        need_inp, need_grid = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if not (need_inp or need_grid):
            return None, None, None, None, None

        grid = grid.detach()
        grad_out = grad_out.detach().contiguous()
        d = grid.shape[-1]
        nbatch = grid.ndim - d - 1
        field_shape = (*grid.shape[:nbatch], *ctx.inshape, ctx.channels)

        if need_grid:
            # One fused call yields both adjoints; `ginp` is scattered into
            # and so must start at zero.
            ginp = grad_out.new_zeros(field_shape)
            ggrid = grad_out.new_zeros(grid.shape)
            _fb.pull_backward(
                ginp,
                ggrid,
                inp.detach(),
                grad_out,
                grid,
                spline=ctx.spline,
                bound=ctx.bound,
                extrapolate=ctx.extrapolate,
                stream=stream_ptr(ginp),
            )
            return (ginp if need_inp else None), ggrid, None, None, None

        # Field-only: the adjoint of `pull` is plain `push` (cheaper -- it
        # never touches the field to compute a spatial gradient).
        ginp = grad_out.new_zeros(field_shape)
        _fb.push(
            ginp,
            grad_out,
            grid,
            spline=ctx.spline,
            bound=ctx.bound,
            extrapolate=ctx.extrapolate,
            stream=stream_ptr(ginp),
        )
        return ginp, None, None, None, None


class _Push(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, grid, spline, bound, extrapolate, spatial):
        d = grid.shape[-1]
        nbatch = grid.ndim - d - 1
        out = inp.new_zeros((*grid.shape[:nbatch], *spatial, inp.shape[-1]))
        _fb.push(
            out,
            inp,
            grid,
            spline=spline,
            bound=bound,
            extrapolate=extrapolate,
            stream=stream_ptr(out),
        )
        ctx.save_for_backward(inp if grid.requires_grad else None, grid)
        ctx.spline, ctx.bound, ctx.extrapolate = spline, bound, extrapolate
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        inp, grid = ctx.saved_tensors
        need_inp, need_grid = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if not (need_inp or need_grid):
            return None, None, None, None, None, None

        grid = grid.detach()
        grad_out = grad_out.detach().contiguous()
        ginp_shape = (*grid.shape[:-1], grad_out.shape[-1])

        if need_grid:
            ginp = grad_out.new_zeros(ginp_shape)
            ggrid = grad_out.new_zeros(grid.shape)
            _fb.push_backward(
                ginp,
                ggrid,
                inp.detach(),
                grad_out,
                grid,
                spline=ctx.spline,
                bound=ctx.bound,
                extrapolate=ctx.extrapolate,
                stream=stream_ptr(ginp),
            )
            return (ginp if need_inp else None), ggrid, None, None, None, None

        # Field-only: the adjoint of `push` is plain `pull`.
        ginp = grad_out.new_zeros(ginp_shape)
        _fb.pull(
            ginp,
            grad_out,
            grid,
            spline=ctx.spline,
            bound=ctx.bound,
            extrapolate=ctx.extrapolate,
            stream=stream_ptr(ginp),
        )
        return ginp, None, None, None, None, None


class _Count(torch.autograd.Function):
    """``out = count(grid)``; differentiable wrt ``grid``.

    ``grid`` is the only input -- the splatted values are implicitly ones --
    so the single adjoint is ``count_backward``, which overwrites its ``gout``
    (one grid point maps to exactly one output element).
    """

    @staticmethod
    def forward(ctx, grid, spline, bound, extrapolate, spatial):
        d = grid.shape[-1]
        nbatch = grid.ndim - d - 1
        out = grid.new_zeros((*grid.shape[:nbatch], *spatial, 1))
        _fb.count(
            out,
            grid,
            spline=spline,
            bound=bound,
            extrapolate=extrapolate,
            stream=stream_ptr(out),
        )
        ctx.save_for_backward(grid)
        ctx.spline, ctx.bound, ctx.extrapolate = spline, bound, extrapolate
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        (grid,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        grid = grid.detach()
        ggrid = grad_out.new_zeros(grid.shape)
        _fb.count_backward(
            ggrid,
            grad_out.detach().contiguous(),
            grid,
            spline=ctx.spline,
            bound=ctx.bound,
            extrapolate=ctx.extrapolate,
            stream=stream_ptr(ggrid),
        )
        return ggrid, None, None, None, None


class _Grad(torch.autograd.Function):
    """``out = grad(inp, grid)``; differentiable wrt ``inp`` and ``grid``.

    ``grad`` samples the *spatial gradient* of the field, so its adjoint
    produces second-order terms -- ``grad_backward`` scatters into the field
    gradient (which must therefore be pre-zeroed) and overwrites the grid
    gradient. ``abs`` must match the forward call, which is why it is saved.
    """

    @staticmethod
    def forward(ctx, inp, grid, spline, bound, extrapolate, abs_):
        d = grid.shape[-1]
        out = inp.new_zeros((*grid.shape[:-1], inp.shape[-1], d))
        _fb.grad(
            out,
            inp,
            grid,
            spline=spline,
            bound=bound,
            extrapolate=extrapolate,
            abs=abs_,
            stream=stream_ptr(out),
        )
        ctx.save_for_backward(inp, grid)
        ctx.spline, ctx.bound, ctx.extrapolate = spline, bound, extrapolate
        ctx.abs_ = abs_
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        inp, grid = ctx.saved_tensors
        need_inp, need_grid = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if not (need_inp or need_grid):
            return None, None, None, None, None, None
        inp, grid = inp.detach(), grid.detach()
        # grad_backward writes both gradients in one pass; pre-zero `ginp`
        # because it scatters into it.
        ginp = grad_out.new_zeros(inp.shape)
        ggrid = grad_out.new_zeros(grid.shape)
        _fb.grad_backward(
            ginp,
            ggrid,
            inp,
            grad_out.detach().contiguous(),
            grid,
            spline=ctx.spline,
            bound=ctx.bound,
            extrapolate=ctx.extrapolate,
            abs=ctx.abs_,
            stream=stream_ptr(ginp),
        )
        return (
            (ginp if need_inp else None),
            (ggrid if need_grid else None),
            None,
            None,
            None,
            None,
        )


def pull(
    inp: Tensor,
    grid: Tensor,
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
) -> Tensor:
    """Sample a spline volume at arbitrary coordinates (gather).

    Coordinates are in voxels, channels come last and the first coordinate is
    x. For a spatial rank `D` of 1, 2 or 3 and any number of leading batch axes
    `B`, the volume `inp` is `(*B, *inshape, C)` and the coordinates `grid` are
    `(*B, *outshape, D)`; `D` is read from the last axis of `grid`.

    `inp` holds spline coefficients, which
    [`spline_coeff`][fastfields.torch.spline_coeff] computes from samples.
    [`push`][fastfields.torch.push] is the adjoint operation.

    Differentiable with respect to both `inp` and `grid`, which is what makes
    it usable inside a learned deformation model.

    Parameters
    ----------
    inp : torch.Tensor
        Volume of spline coefficients, shape `(*B, *inshape, C)`.
    grid : torch.Tensor
        Sampling coordinates in voxels, x first, shape `(*B, *outshape,
        D)`. `D` is the spatial rank (1, 2 or 3) and `grid` must have the
        same number of axes as `inp`.
    order : int or str, default=2
        Spline order: an integer from 0 to 7, a
        [`Spline`][fastfields.helpers.Spline] member, or a name such as
        `"cubic"`.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    extrapolate : int, default=1
        How far outside the field of view samples still count: `1` always,
        `0` not past the voxel centres, `-1` not past the voxel edges.

    Returns
    -------
    torch.Tensor
        The sampled values, shape `(*B, *outshape, C)`.
    """
    check_dtype(inp, grid)
    if grid.ndim != inp.ndim:
        raise ValueError("inp and grid must have the same rank")
    return _Pull.apply(
        inp, grid, as_spline(order), as_bound(bound), int(extrapolate)
    )


def push(
    inp: Tensor,
    grid: Tensor,
    shape: int | Sequence[int],
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
) -> Tensor:
    """Splat values into a spline volume (scatter).

    Coordinates are in voxels, channels come last and the first coordinate is
    x. For a spatial rank `D` of 1, 2 or 3 and any number of leading batch axes
    `B`, the volume `inp` is `(*B, *inshape, C)` and the coordinates `grid` are
    `(*B, *outshape, D)`; `D` is read from the last axis of `grid`. Here `inp`
    holds one value per coordinate, `(*B, *outshape, C)`, and the result is a
    volume `(*B, *shape, C)`.

    The adjoint of [`pull`][fastfields.torch.pull]. Values landing on the same
    voxel are summed; the result starts from zero.

    Differentiable with respect to both `inp` and `grid`.

    Parameters
    ----------
    inp : torch.Tensor
        Values to splat, shape `(*B, *outshape, C)`, one per coordinate in
        `grid`.
    grid : torch.Tensor
        Sampling coordinates in voxels, x first, shape `(*B, *outshape,
        D)`. `D` is the spatial rank (1, 2 or 3) and `grid` must have the
        same number of axes as `inp`.
    shape : int or sequence of int
        Spatial size of the target volume, one value per spatial axis or a
        single value for all.
    order : int or str, default=2
        Spline order: an integer from 0 to 7, a
        [`Spline`][fastfields.helpers.Spline] member, or a name such as
        `"cubic"`.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    extrapolate : int, default=1
        How far outside the field of view samples still count: `1` always,
        `0` not past the voxel centres, `-1` not past the voxel edges.

    Returns
    -------
    torch.Tensor
        The volume, shape `(*B, *shape, C)`.
    """
    check_dtype(inp, grid)
    if grid.ndim != inp.ndim:
        raise ValueError("inp and grid must have the same rank")
    spatial = _spatial(shape, grid.shape[-1])
    return _Push.apply(
        inp, grid, as_spline(order), as_bound(bound), int(extrapolate), spatial
    )


def count(
    grid: Tensor,
    shape: int | Sequence[int],
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
) -> Tensor:
    """Splat ones into a volume: how many samples land on each voxel.

    Coordinates are in voxels, channels come last and the first coordinate is
    x. The coordinates `grid` are `(*B, *outshape, D)`; `D` is read from their
    last axis.

    Equivalent to [`push`][fastfields.torch.push] of an all-ones input: the
    weight each voxel receives from the sampling pattern.

    Differentiable with respect to `grid`, its only input.

    Parameters
    ----------
    grid : torch.Tensor
        Sampling coordinates in voxels, x first, shape `(*B, *outshape,
        D)`; `D` is the spatial rank (1, 2 or 3).
    shape : int or sequence of int
        Spatial size of the target volume, one value per spatial axis or a
        single value for all.
    order : int or str, default=2
        Spline order: an integer from 0 to 7, a
        [`Spline`][fastfields.helpers.Spline] member, or a name such as
        `"cubic"`.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    extrapolate : int, default=1
        How far outside the field of view samples still count: `1` always,
        `0` not past the voxel centres, `-1` not past the voxel edges.

    Returns
    -------
    torch.Tensor
        The weights, shape `(*B, *shape, 1)`.
    """
    check_dtype(grid)
    spatial = _spatial(shape, grid.shape[-1])
    return _Count.apply(
        grid, as_spline(order), as_bound(bound), int(extrapolate), spatial
    )


def grad(
    inp: Tensor,
    grid: Tensor,
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
    abs: bool = False,
) -> Tensor:
    """Sample the spatial gradient of a spline volume at arbitrary coordinates.

    Coordinates are in voxels, channels come last and the first coordinate is
    x. For a spatial rank `D` of 1, 2 or 3 and any number of leading batch axes
    `B`, the volume `inp` is `(*B, *inshape, C)` and the coordinates `grid` are
    `(*B, *outshape, D)`; `D` is read from the last axis of `grid`. The result
    holds, for every coordinate and channel, the vector of the `D` spatial
    derivatives.

    With `abs=True` the absolute value of each spline derivative is used
    instead of its signed value.

    Differentiable with respect to both `inp` and `grid`. Because `grad` is
    itself a derivative operator, differentiating it yields second-order terms;
    that is what lets an image gradient computed inside a loss train the
    deformation that produced `grid`.

    With `abs=True` the backward pass is the adjoint of the absolute-value
    operator, which is a majorisation and deliberately not the Jacobian of the
    forward map.

    Parameters
    ----------
    inp : torch.Tensor
        Volume of spline coefficients, shape `(*B, *inshape, C)`.
    grid : torch.Tensor
        Sampling coordinates in voxels, x first, shape `(*B, *outshape,
        D)`. `D` is the spatial rank (1, 2 or 3) and `grid` must have the
        same number of axes as `inp`.
    order : int or str, default=2
        Spline order: an integer from 0 to 7, a
        [`Spline`][fastfields.helpers.Spline] member, or a name such as
        `"cubic"`.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    extrapolate : int, default=1
        How far outside the field of view samples still count: `1` always,
        `0` not past the voxel centres, `-1` not past the voxel edges.
    abs : bool, default=False
        Take the absolute value of each spline derivative.

    Returns
    -------
    torch.Tensor
        The gradients, shape `(*B, *outshape, C, D)`.
    """
    check_dtype(inp, grid)
    if grid.ndim != inp.ndim:
        raise ValueError("inp and grid must have the same rank")
    return _Grad.apply(
        inp,
        grid,
        as_spline(order),
        as_bound(bound),
        int(extrapolate),
        bool(abs),
    )
