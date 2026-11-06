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
    """Spline coefficients of a signal, along the last axis.

    Converts samples into the coefficients of the interpolating spline of order
    `order`, so that evaluating that spline at the sample positions gives the
    samples back. For orders 0 and 1 the coefficients are the samples. Only the
    last axis is filtered.

    For orders above 1 only the boundary conditions `dct1`, `dct2`, `dft` and
    `replicate` are implemented; any other is rejected. The coefficients are
    what [`pull`][fastfields.numpy.pull], [`push`][fastfields.numpy.push] and
    [`grad`][fastfields.numpy.grad] expect as their volume.

    [`spline_coeff_`][fastfields.numpy.spline_coeff_] is the in-place version.

    Parameters
    ----------
    inp : numpy.ndarray
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
    numpy.ndarray
        The coefficients, a new array the shape of `inp`.
    """
    out = _as_float_array(inp, "inp", copy=True)
    _ff.spline_coeff(out, _as_spline(order), _as_bound(bound))
    return out


def spline_coeff_(
    inp_out: np.ndarray, order: int | str = 3, bound: int | str = "dct2"
) -> np.ndarray:
    """Spline coefficients of a signal along the last axis, in place.

    The in-place version of [`spline_coeff`][fastfields.numpy.spline_coeff]:
    the samples in `inp_out` are replaced by the coefficients. For orders above
    1 only the boundary conditions `dct1`, `dct2`, `dft` and `replicate` are
    implemented; any other is rejected.

    Parameters
    ----------
    inp_out : numpy.ndarray
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
    numpy.ndarray
        `inp_out` itself, now holding the coefficients.

    Raises
    ------
    TypeError
        If `inp_out` is not a float32 or float64 NumPy array.
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
    """Resample the trailing axes of an array with splines.

    Resamples `inp` onto a new grid with splines of order `order`, along the
    last `ndim` axes; leading axes are batch axes. Give either a size `factor`
    or an explicit output `shape`, not both.

    Which points of the two grids line up is set by `anchor`; with `"centers"`,
    the default, the first and last samples coincide. The `anchor` determines
    the sampling `scale` and `shift` for the kernel, and either can be
    overridden.

    [`restriction`][fastfields.numpy.restriction] is the adjoint operation.

    !!! tip "Interpolating resize"
        `resample` does not prefilter its input. For a true interpolating
        resize, pass the coefficients computed by
        [`spline_coeff`][fastfields.numpy.spline_coeff] instead of the raw
        samples.

    Parameters
    ----------
    inp : numpy.ndarray
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
    numpy.ndarray
        The resampled array, shape `(*batch, *spatial_out)`.

    Raises
    ------
    ValueError
        If `ndim` is outside `1..inp.ndim`, `anchor` is unknown, or
        `factor`, `shape` or `scale` has the wrong length.
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
    """Adjoint of `resample`, along the trailing axes of an array.

    Transposes the linear map of [`resample`][fastfields.numpy.resample] for
    the same arguments: values on the finer grid are accumulated onto the
    coarser one. The `scale` is derived from this call's own input and output
    shapes, so a `resample` and a `restriction` with matching shapes use
    reciprocal scales and the same shift.

    The binding accumulates into its output buffer, which is why the buffer is
    allocated zero-filled here. When calling `fastfields.dlpack.restriction`
    yourself, pre-zero `out`.

    `factor`, `shape`, `order`, `bound`, `ndim`, `anchor`, `shift` and `scale`
    are exactly as for [`resample`][fastfields.numpy.resample].

    Parameters
    ----------
    inp : numpy.ndarray
        Input, shape `(*batch, *spatial_in)`; the last `ndim` axes are
        resampled.
    factor : float or sequence of float, optional
        Size multiplier, as for [`resample`][fastfields.numpy.resample].
        Exclusive with `shape`.
    shape : int or sequence of int, optional
        Output spatial size, as for
        [`resample`][fastfields.numpy.resample]. Exclusive with `factor`.
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
        [`resample`][fastfields.numpy.resample].
    anchor : {"centers", "edges", "first", "last"}, default="centers"
        Grid alignment, as for [`resample`][fastfields.numpy.resample].
    shift : float, optional
        Sampling shift, as for [`resample`][fastfields.numpy.resample].
    scale : sequence of float, optional
        Per-axis scale, as for [`resample`][fastfields.numpy.resample].

    Returns
    -------
    numpy.ndarray
        The restricted array, shape `(*batch, *spatial_out)`.

    Raises
    ------
    ValueError
        If `ndim` is outside `1..inp.ndim`, `anchor` is unknown, or
        `factor`, `shape` or `scale` has the wrong length.
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
