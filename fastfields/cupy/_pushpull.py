"""Spline sampling (pushpull) for CuPy.

The same layout and signatures as `fastfields.numpy`: coordinates in voxels,
channels last, x first, with `D` read from the last axis of `grid`.
"""

from __future__ import annotations

from typing import Any, Sequence

import fastfields.dlpack as _ff

from fastfields.helpers import as_bound, as_spline

from ._util import as_gpu_array, cupy, current_stream_ptr

__all__ = ["pull", "push", "count", "grad"]


def _spatial(shape: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,) * ndim
    out = tuple(int(s) for s in shape)
    if len(out) != ndim:
        raise ValueError(f"shape must have length ndim={ndim}, got {shape!r}")
    return out


def pull(
    inp: Any,
    grid: Any,
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
) -> Any:
    """Sample a spline volume at arbitrary coordinates (gather).

    Coordinates are in voxels, channels come last and the first coordinate is
    x. For a spatial rank `D` of 1, 2 or 3 and any number of leading batch axes
    `B`, the volume `inp` is `(*B, *inshape, C)` and the coordinates `grid` are
    `(*B, *outshape, D)`; `D` is read from the last axis of `grid`.

    `inp` holds spline coefficients, which
    [`spline_coeff`][fastfields.cupy.spline_coeff] computes from samples.
    [`push`][fastfields.cupy.push] is the adjoint operation.

    Parameters
    ----------
    inp : cupy.ndarray
        Volume of spline coefficients, shape `(*B, *inshape, C)`.
    grid : cupy.ndarray
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
    cupy.ndarray
        The sampled values, shape `(*B, *outshape, C)`.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    cp = cupy()
    inp = as_gpu_array(inp, name="inp")
    grid = as_gpu_array(grid, name="grid")
    if grid.ndim != inp.ndim:
        raise ValueError("inp and grid must have the same rank")
    out = cp.zeros((*grid.shape[:-1], inp.shape[-1]), dtype=inp.dtype)
    _ff.pull(
        out,
        inp,
        grid,
        as_spline(order),
        as_bound(bound),
        int(extrapolate),
        current_stream_ptr(),
    )
    return out


def push(
    inp: Any,
    grid: Any,
    shape: int | Sequence[int],
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
) -> Any:
    """Splat values into a spline volume (scatter).

    Coordinates are in voxels, channels come last and the first coordinate is
    x. For a spatial rank `D` of 1, 2 or 3 and any number of leading batch axes
    `B`, the volume `inp` is `(*B, *inshape, C)` and the coordinates `grid` are
    `(*B, *outshape, D)`; `D` is read from the last axis of `grid`. Here `inp`
    holds one value per coordinate, `(*B, *outshape, C)`, and the result is a
    volume `(*B, *shape, C)`.

    The adjoint of [`pull`][fastfields.cupy.pull]. Values landing on the same
    voxel are summed; the result starts from zero.

    Parameters
    ----------
    inp : cupy.ndarray
        Values to splat, shape `(*B, *outshape, C)`, one per coordinate in
        `grid`.
    grid : cupy.ndarray
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
    cupy.ndarray
        The volume, shape `(*B, *shape, C)`.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    cp = cupy()
    inp = as_gpu_array(inp, name="inp")
    grid = as_gpu_array(grid, name="grid")
    if grid.ndim != inp.ndim:
        raise ValueError("inp and grid must have the same rank")
    ndim = grid.shape[-1]
    nbatch = grid.ndim - ndim - 1
    if nbatch < 0:
        raise ValueError("grid rank is too small for the coordinate dim")
    spatial = _spatial(shape, ndim)
    out = cp.zeros(
        (*grid.shape[:nbatch], *spatial, inp.shape[-1]), dtype=inp.dtype
    )
    _ff.push(
        out,
        inp,
        grid,
        as_spline(order),
        as_bound(bound),
        int(extrapolate),
        current_stream_ptr(),
    )
    return out


def count(
    grid: Any,
    shape: int | Sequence[int],
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
) -> Any:
    """Splat ones into a volume: how many samples land on each voxel.

    Coordinates are in voxels, channels come last and the first coordinate is
    x. The coordinates `grid` are `(*B, *outshape, D)`; `D` is read from their
    last axis.

    Equivalent to [`push`][fastfields.cupy.push] of an all-ones input: the
    weight each voxel receives from the sampling pattern.

    Parameters
    ----------
    grid : cupy.ndarray
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
    cupy.ndarray
        The weights, shape `(*B, *shape, 1)`.

    Raises
    ------
    TypeError
        If `grid` is not a float32 or float64 CuPy array.
    """
    cp = cupy()
    grid = as_gpu_array(grid, name="grid")
    ndim = grid.shape[-1]
    nbatch = grid.ndim - ndim - 1
    if nbatch < 0:
        raise ValueError("grid rank is too small for the coordinate dim")
    spatial = _spatial(shape, ndim)
    out = cp.zeros((*grid.shape[:nbatch], *spatial, 1), dtype=grid.dtype)
    _ff.count(
        out,
        grid,
        as_spline(order),
        as_bound(bound),
        int(extrapolate),
        current_stream_ptr(),
    )
    return out


def grad(
    inp: Any,
    grid: Any,
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
    abs: bool = False,
) -> Any:
    """Sample the spatial gradient of a spline volume at arbitrary coordinates.

    Coordinates are in voxels, channels come last and the first coordinate is
    x. For a spatial rank `D` of 1, 2 or 3 and any number of leading batch axes
    `B`, the volume `inp` is `(*B, *inshape, C)` and the coordinates `grid` are
    `(*B, *outshape, D)`; `D` is read from the last axis of `grid`. The result
    holds, for every coordinate and channel, the vector of the `D` spatial
    derivatives.

    With `abs=True` the absolute value of each spline derivative is used
    instead of its signed value.

    Parameters
    ----------
    inp : cupy.ndarray
        Volume of spline coefficients, shape `(*B, *inshape, C)`.
    grid : cupy.ndarray
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
    cupy.ndarray
        The gradients, shape `(*B, *outshape, C, D)`.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    cp = cupy()
    inp = as_gpu_array(inp, name="inp")
    grid = as_gpu_array(grid, name="grid")
    if grid.ndim != inp.ndim:
        raise ValueError("inp and grid must have the same rank")
    ndim = grid.shape[-1]
    out = cp.zeros((*grid.shape[:-1], inp.shape[-1], ndim), dtype=inp.dtype)
    _ff.grad(
        out,
        inp,
        grid,
        as_spline(order),
        as_bound(bound),
        int(extrapolate),
        bool(abs),
        current_stream_ptr(),
    )
    return out
