"""Spline-coefficient prefilter and resampling/restriction (numpy)."""

from __future__ import annotations

from typing import Sequence

import fastfields.dlpack as _ff

import numpy as np
from fastfields.helpers import (
    anchor_scale_shift,
    infer_ndim,
    resolve_out_spatial,
)

from ._util import _as_bound, _as_float_array, _as_spline, _validate_inplace

__all__ = [
    "spline_coeff",
    "spline_coeff_",
    "resample",
    "restriction",
]


def spline_coeff(
    inp: np.ndarray, order: int | str = 3, bound: int | str = "dct2"
) -> np.ndarray:
    """Spline-coefficient prefilter along the **last** axis.

    Parameters
    ----------
    inp : numpy.ndarray
        Input samples.
    order : int or str, default=3
        Spline order (orders 0 and 1 are no-ops).
    bound : int or str, default="dct2"
        Boundary condition.

    Returns
    -------
    numpy.ndarray
        Spline coefficients, a newly allocated array; ``inp`` is left
        untouched. See [`spline_coeff_`][fastfields.numpy.spline_coeff_] for
        the in-place variant.
    """
    out = _as_float_array(inp, "inp", copy=True)
    _ff.spline_coeff(out, _as_spline(order), _as_bound(bound))
    return out


def spline_coeff_(
    inp_out: np.ndarray, order: int | str = 3, bound: int | str = "dct2"
) -> np.ndarray:
    """In-place spline-coefficient prefilter along the **last** axis.

    Parameters
    ----------
    inp_out : numpy.ndarray
        Input samples. Must be a float32/float64 array; mutated in place and
        returned.
    order : int or str, default=3
        Spline order (orders 0 and 1 are no-ops).
    bound : int or str, default="dct2"
        Boundary condition.

    Returns
    -------
    numpy.ndarray
        ``inp_out`` (the same array object), now holding the spline
        coefficients.
    """
    _validate_inplace(inp_out, "inp_out")
    _ff.spline_coeff(inp_out, _as_spline(order), _as_bound(bound))
    return inp_out


def _resize_shapes(
    in_shape: tuple[int, ...],
    ndim: int,
    factor: float | Sequence[float] | None,
    shape: int | Sequence[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return the batch, input-spatial, and output-spatial shapes.

    The output spatial shape is resolved via
    [`fastfields.helpers.resolve_out_spatial`][fastfields.helpers.resolve_out_spatial]
    so every backend shares one implementation. Raises ``ValueError`` if
    ``factor``/``shape`` do not have length ``ndim``.
    """
    batch = tuple(in_shape[:-ndim]) if ndim < len(in_shape) else ()
    spatial_in = tuple(in_shape[-ndim:])
    out_spatial = resolve_out_spatial(spatial_in, ndim, factor, shape)
    return batch, spatial_in, out_spatial


def resample(
    inp: np.ndarray,
    factor: float | Sequence[float] | None = None,
    shape: int | Sequence[int] | None = None,
    *,
    order: int | str = 2,
    bound: int | str = "dct2",
    ndim: int | None = None,
    anchor: str = "centers",
    shift: float | None = None,
    scale: Sequence[float] | None = None,
) -> np.ndarray:
    """Spline resample (prolongation) of the last ``ndim`` axes.

    Parameters
    ----------
    inp : numpy.ndarray
        Input array.
    factor : float or sequence of float, optional
        Per-axis multiplier (scalar or sequence). Mutually exclusive with
        ``shape``; with neither, this is the identity.
    shape : int or sequence of int, optional
        Explicit output spatial size.
    order : int or str, default=2
        Spline order.
    bound : int or str, default="dct2"
        Boundary condition.
    ndim : int, optional
        Number of trailing spatial dimensions (inferred when omitted).
    anchor : {"centers", "edges", "first", "last"}, default="centers"
        Sampling-grid convention, matching ``interpol.resize``. Determines the
        per-dim scale and the default ``shift``:

        * ``centers`` -- align first/last samples (``(in-1)/(out-1)`` step);
        * ``edges`` -- align the outer voxel *edges* (half-voxel shift);
        * ``first`` -- anchor the first voxel (``in/out`` step, no shift);
        * ``last`` -- anchor the last voxel.

        Abbreviations (``"c"``/``"e"``/``"f"``/``"l"``) are accepted.
    shift : float, optional
        Sampling shift override. When omitted the shift implied by ``anchor``
        is used; pass a value to override it (advanced use).
    scale : sequence of float, optional
        Per-dim scale override (default: derived from ``anchor`` and the
        shapes), length ``ndim``.

    Returns
    -------
    numpy.ndarray
        The resampled array.

    Raises
    ------
    ValueError
        If ``ndim`` is outside ``1..inp.ndim`` or ``anchor`` is unknown.
    """
    inp = _as_float_array(inp, "inp")
    ndim = infer_ndim(ndim, factor, shape)
    if ndim < 1 or ndim > inp.ndim:
        raise ValueError(f"ndim must be in 1..{inp.ndim}, got {ndim}")
    batch, spatial_in, out_spatial = _resize_shapes(
        inp.shape, ndim, factor, shape
    )
    eff_scale, anchor_shift = anchor_scale_shift(
        anchor, spatial_in, out_spatial, ndim
    )
    if scale is not None:
        eff_scale = [float(s) for s in scale]
        if len(eff_scale) != ndim:
            raise ValueError(
                f"Expected scale of length ndim={ndim}, got {scale}."
            )
    out = np.zeros(batch + out_spatial, dtype=inp.dtype)
    _ff.resample(
        out,
        inp,
        spline=_as_spline(order),
        bound=_as_bound(bound),
        shift=anchor_shift if shift is None else float(shift),
        scale=eff_scale,
        ndim=ndim,
    )
    return out


def restriction(
    inp: np.ndarray,
    factor: float | Sequence[float] | None = None,
    shape: int | Sequence[int] | None = None,
    *,
    order: int | str = 2,
    bound: int | str = "dct2",
    ndim: int | None = None,
    anchor: str = "centers",
    shift: float | None = None,
    scale: Sequence[float] | None = None,
) -> np.ndarray:
    """Restriction (adjoint of [`resample`][fastfields.numpy.resample]) of the
    last ``ndim`` axes.

    The output buffer is zeroed before the (accumulating) binding call. The
    ``anchor`` convention matches [`resample`][fastfields.numpy.resample];
    because the scale is derived from this call's own (input, output) shapes, a
    ``resample`` and a matching ``restriction`` use reciprocal scales and the
    same shift -- the adjoint relationship the binding expects.

    Parameters
    ----------
    inp : numpy.ndarray
        Input array.
    factor : float or sequence of float, optional
        Per-axis multiplier (scalar or sequence). Mutually exclusive with
        ``shape``.
    shape : int or sequence of int, optional
        Explicit output spatial size.
    order : int or str, default=2
        Spline order.
    bound : int or str, default="dct2"
        Boundary condition.
    ndim : int, optional
        Number of trailing spatial dimensions (inferred when omitted).
    anchor : {"centers", "edges", "first", "last"}, default="centers"
        Sampling-grid convention (see [`resample`][fastfields.numpy.resample]).
    shift : float, optional
        Sampling shift override (see [`resample`][fastfields.numpy.resample]).
    scale : sequence of float, optional
        Per-dim scale override (see [`resample`][fastfields.numpy.resample]).

    Returns
    -------
    numpy.ndarray
        The restricted array.

    Raises
    ------
    ValueError
        If ``ndim`` is outside ``1..inp.ndim`` or ``anchor`` is unknown.
    """
    inp = _as_float_array(inp, "inp")
    ndim = infer_ndim(ndim, factor, shape)
    if ndim < 1 or ndim > inp.ndim:
        raise ValueError(f"ndim must be in 1..{inp.ndim}, got {ndim}")
    batch, spatial_in, out_spatial = _resize_shapes(
        inp.shape, ndim, factor, shape
    )
    eff_scale, anchor_shift = anchor_scale_shift(
        anchor, spatial_in, out_spatial, ndim
    )
    if scale is not None:
        eff_scale = [float(s) for s in scale]
        if len(eff_scale) != ndim:
            raise ValueError(
                f"Expected scale of length ndim={ndim}, got {scale}."
            )
    # pre-zeroed (the binding accumulates into `out`)
    out = np.zeros(batch + out_spatial, dtype=inp.dtype)
    _ff.restriction(
        out,
        inp,
        spline=_as_spline(order),
        bound=_as_bound(bound),
        shift=anchor_shift if shift is None else float(shift),
        scale=eff_scale,
        ndim=ndim,
    )
    return out
