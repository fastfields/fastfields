"""Spline sampling (pushpull) for NumPy.

Coordinates are in voxels, channels come last and the first coordinate is x.
For a spatial rank `D` of 1, 2 or 3 and leading batch axes `B`:

- `inp` has shape `(*B, *inshape, C)` and holds spline coefficients;
- `grid` has shape `(*B, *outshape, D)`;
- `pull` returns `(*B, *outshape, C)`.

`inp` and `grid` must have the same number of axes; `D` is read from the last
axis of `grid`. `push` is the adjoint of `pull`.
"""

from __future__ import annotations

from typing import Sequence

import fastfields.dlpack as _ff

import numpy as np

from ._util import _as_bound, _as_float_array, _as_spline

__all__ = ["pull", "push", "count", "grad"]


def _spatial_shape(shape: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    """Normalise a spatial-size argument to a length-``ndim`` tuple."""
    if np.isscalar(shape):
        return (int(shape),) * ndim
    out = tuple(int(s) for s in shape)
    if len(out) != ndim:
        raise ValueError(f"shape must have length ndim={ndim}, got {shape!r}")
    return out


def pull(
    inp: np.ndarray,
    grid: np.ndarray,
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
) -> np.ndarray:
    """Sample a spline volume at arbitrary coordinates (gather).

    Coordinates are in voxels, channels come last and the first coordinate is
    x. For a spatial rank `D` of 1, 2 or 3 and any number of leading batch axes
    `B`, the volume `inp` is `(*B, *inshape, C)` and the coordinates `grid` are
    `(*B, *outshape, D)`; `D` is read from the last axis of `grid`.

    `inp` holds spline coefficients, which
    [`spline_coeff`][fastfields.numpy.spline_coeff] computes from samples.
    [`push`][fastfields.numpy.push] is the adjoint operation.

    Parameters
    ----------
    inp : numpy.ndarray
        Volume of spline coefficients, shape `(*B, *inshape, C)`.
    grid : numpy.ndarray
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
    numpy.ndarray
        The sampled values, shape `(*B, *outshape, C)`.

    Raises
    ------
    ValueError
        If `inp` and `grid` do not have the same number of axes.
    """
    inp = _as_float_array(inp, "inp")
    grid = _as_float_array(grid, "grid", dtype=inp.dtype)
    if grid.ndim != inp.ndim:
        raise ValueError("inp and grid must have the same rank")
    out = np.zeros(grid.shape[:-1] + (inp.shape[-1],), dtype=inp.dtype)
    _ff.pull(
        out,
        inp,
        grid,
        spline=_as_spline(order),
        bound=_as_bound(bound),
        extrapolate=int(extrapolate),
    )
    return out


def push(
    inp: np.ndarray,
    grid: np.ndarray,
    shape: int | Sequence[int],
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
) -> np.ndarray:
    """Splat values into a spline volume (scatter).

    Coordinates are in voxels, channels come last and the first coordinate is
    x. For a spatial rank `D` of 1, 2 or 3 and any number of leading batch axes
    `B`, the volume `inp` is `(*B, *inshape, C)` and the coordinates `grid` are
    `(*B, *outshape, D)`; `D` is read from the last axis of `grid`. Here `inp`
    holds one value per coordinate, `(*B, *outshape, C)`, and the result is a
    volume `(*B, *shape, C)`.

    The adjoint of [`pull`][fastfields.numpy.pull]. Values landing on the same
    voxel are summed; the result starts from zero.

    Parameters
    ----------
    inp : numpy.ndarray
        Values to splat, shape `(*B, *outshape, C)`, one per coordinate in
        `grid`.
    grid : numpy.ndarray
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
    numpy.ndarray
        The volume, shape `(*B, *shape, C)`.

    Raises
    ------
    ValueError
        If `inp` and `grid` do not have the same number of axes, or `shape`
        has the wrong length.
    """
    inp = _as_float_array(inp, "inp")
    grid = _as_float_array(grid, "grid", dtype=inp.dtype)
    if grid.ndim != inp.ndim:
        raise ValueError("inp and grid must have the same rank")
    ndim = grid.shape[-1]
    nbatch = grid.ndim - ndim - 1
    if nbatch < 0:
        raise ValueError("grid rank is too small for the coordinate dim")
    spatial = _spatial_shape(shape, ndim)
    out = np.zeros(
        grid.shape[:nbatch] + spatial + (inp.shape[-1],), dtype=inp.dtype
    )
    _ff.push(
        out,
        inp,
        grid,
        spline=_as_spline(order),
        bound=_as_bound(bound),
        extrapolate=int(extrapolate),
    )
    return out


def count(
    grid: np.ndarray,
    shape: int | Sequence[int],
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
) -> np.ndarray:
    """Splat ones into a volume: how many samples land on each voxel.

    Coordinates are in voxels, channels come last and the first coordinate is
    x. The coordinates `grid` are `(*B, *outshape, D)`; `D` is read from their
    last axis.

    Equivalent to [`push`][fastfields.numpy.push] of an all-ones input: the
    weight each voxel receives from the sampling pattern.

    Parameters
    ----------
    grid : numpy.ndarray
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
    numpy.ndarray
        The weights, shape `(*B, *shape, 1)`.
    """
    grid = _as_float_array(grid, "grid")
    ndim = grid.shape[-1]
    nbatch = grid.ndim - ndim - 1
    if nbatch < 0:
        raise ValueError("grid rank is too small for the coordinate dim")
    spatial = _spatial_shape(shape, ndim)
    out = np.zeros(grid.shape[:nbatch] + spatial + (1,), dtype=grid.dtype)
    _ff.count(
        out,
        grid,
        spline=_as_spline(order),
        bound=_as_bound(bound),
        extrapolate=int(extrapolate),
    )
    return out


def grad(
    inp: np.ndarray,
    grid: np.ndarray,
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
    abs: bool = False,
) -> np.ndarray:
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
    inp : numpy.ndarray
        Volume of spline coefficients, shape `(*B, *inshape, C)`.
    grid : numpy.ndarray
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
    numpy.ndarray
        The gradients, shape `(*B, *outshape, C, D)`.

    Raises
    ------
    ValueError
        If `inp` and `grid` do not have the same number of axes.
    """
    inp = _as_float_array(inp, "inp")
    grid = _as_float_array(grid, "grid", dtype=inp.dtype)
    if grid.ndim != inp.ndim:
        raise ValueError("inp and grid must have the same rank")
    ndim = grid.shape[-1]
    out = np.zeros(grid.shape[:-1] + (inp.shape[-1], ndim), dtype=inp.dtype)
    _ff.grad(
        out,
        inp,
        grid,
        spline=_as_spline(order),
        bound=_as_bound(bound),
        extrapolate=int(extrapolate),
        abs=bool(abs),
    )
    return out
