"""Resample / restriction / spline-coefficient wrappers (cupy)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import fastfields.dlpack as _ff

from fastfields.helpers import (
    anchor_scale_shift,
    as_bound,
    as_spline,
    check_ndim,
    infer_ndim,
    resolve_out_spatial,
)

from ._util import (
    as_gpu_array,
    cupy,
    current_stream_ptr,
    require_gpu_writethrough,
)

__all__ = [
    "resample",
    "restriction",
    "spline_coeff",
    "spline_coeff_",
]


def _resolve(
    inp: Any,
    factor: float | Sequence[float] | None,
    shape: int | Sequence[int] | None,
    ndim: Optional[int],
    anchor: str,
    scale: Optional[Sequence[float]],
    shift: Optional[float],
) -> tuple[tuple[int, ...], list[float], float]:
    """Resolve (full output shape, per-dim scale, scalar shift) for a call.

    The order/bound normalisation and the anchor/factor/shape resolution come
    from [`fastfields.helpers`][fastfields.helpers] so every backend shares one
    implementation. Raises ``ValueError`` if ``ndim`` is outside
    ``1..inp.ndim`` or an explicit ``scale`` has the wrong length.
    """
    ndim = infer_ndim(ndim, factor, shape)
    check_ndim(ndim, inp.ndim)
    spatial_in = tuple(inp.shape[-ndim:])
    out_spatial = resolve_out_spatial(spatial_in, ndim, factor, shape)
    a_scale, a_shift = anchor_scale_shift(
        anchor, spatial_in, out_spatial, ndim
    )
    if scale is not None:
        a_scale = [float(s) for s in scale]
        if len(a_scale) != ndim:
            raise ValueError(
                f"Expected scale of length ndim={ndim}, got {scale}."
            )
    if shift is not None:
        a_shift = float(shift)
    out_shape = tuple(int(n) for n in inp.shape[:-ndim]) + out_spatial
    return out_shape, a_scale, a_shift


def resample(
    inp: Any,
    factor: float | Sequence[float] | None = None,
    shape: int | Sequence[int] | None = None,
    *,
    order: int | str = 2,
    bound: int | str = "dct2",
    ndim: Optional[int] = None,
    anchor: str = "centers",
    shift: Optional[float] = None,
    scale: Optional[Sequence[float]] = None,
) -> Any:
    """Resample the trailing axes of an array with splines.

    Resamples `inp` onto a new grid with splines of order `order`, along the
    last `ndim` axes; leading axes are batch axes. Give either a size `factor`
    or an explicit output `shape`, not both.

    Which points of the two grids line up is set by `anchor`; with `"centers"`,
    the default, the first and last samples coincide. The `anchor` determines
    the sampling `scale` and `shift` for the kernel, and either can be
    overridden.

    [`restriction`][fastfields.cupy.restriction] is the adjoint operation.

    !!! tip "Interpolating resize"
        `resample` does not prefilter its input. For a true interpolating
        resize, pass the coefficients computed by
        [`spline_coeff`][fastfields.cupy.spline_coeff] instead of the raw
        samples.

    Parameters
    ----------
    inp : cupy.ndarray
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
    cupy.ndarray
        The resampled array, shape `(*batch, *spatial_out)`.

    Raises
    ------
    ValueError
        If `ndim` is outside `1..inp.ndim`, `anchor` is unknown, or
        `factor`, `shape` or `scale` has the wrong length.
    TypeError
        If `inp` is not a float32 or float64 CuPy array.
    """
    cp = cupy()
    inp = as_gpu_array(inp, name="inp")
    out_shape, scale, shift = _resolve(
        inp, factor, shape, ndim, anchor, scale, shift
    )
    ndim = len(scale)
    out = cp.empty(out_shape, dtype=inp.dtype)
    _ff.resample(
        out,
        inp,
        as_spline(order),
        as_bound(bound),
        shift,
        scale,
        ndim,
        current_stream_ptr(),
    )
    return out


def restriction(
    inp: Any,
    factor: float | Sequence[float] | None = None,
    shape: int | Sequence[int] | None = None,
    *,
    order: int | str = 2,
    bound: int | str = "dct2",
    ndim: Optional[int] = None,
    anchor: str = "centers",
    shift: Optional[float] = None,
    scale: Optional[Sequence[float]] = None,
) -> Any:
    """Adjoint of `resample`, along the trailing axes of an array.

    Transposes the linear map of [`resample`][fastfields.cupy.resample] for the
    same arguments: values on the finer grid are accumulated onto the coarser
    one. The `scale` is derived from this call's own input and output shapes,
    so a `resample` and a `restriction` with matching shapes use reciprocal
    scales and the same shift.

    The binding accumulates into its output buffer, which is why the buffer is
    allocated zero-filled here. When calling `fastfields.dlpack.restriction`
    yourself, pre-zero `out`.

    `factor`, `shape`, `order`, `bound`, `ndim`, `anchor`, `shift` and `scale`
    are exactly as for [`resample`][fastfields.cupy.resample].

    Parameters
    ----------
    inp : cupy.ndarray
        Input, shape `(*batch, *spatial_in)`; the last `ndim` axes are
        resampled.
    factor : float or sequence of float, optional
        Size multiplier, as for [`resample`][fastfields.cupy.resample].
        Exclusive with `shape`.
    shape : int or sequence of int, optional
        Output spatial size, as for [`resample`][fastfields.cupy.resample].
        Exclusive with `factor`.
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
        [`resample`][fastfields.cupy.resample].
    anchor : {"centers", "edges", "first", "last"}, default="centers"
        Grid alignment, as for [`resample`][fastfields.cupy.resample].
    shift : float, optional
        Sampling shift, as for [`resample`][fastfields.cupy.resample].
    scale : sequence of float, optional
        Per-axis scale, as for [`resample`][fastfields.cupy.resample].

    Returns
    -------
    cupy.ndarray
        The restricted array, shape `(*batch, *spatial_out)`.

    Raises
    ------
    ValueError
        If `ndim` is outside `1..inp.ndim`, `anchor` is unknown, or
        `factor`, `shape` or `scale` has the wrong length.
    TypeError
        If `inp` is not a float32 or float64 CuPy array.
    """
    cp = cupy()
    inp = as_gpu_array(inp, name="inp")
    out_shape, scale, shift = _resolve(
        inp, factor, shape, ndim, anchor, scale, shift
    )
    ndim = len(scale)
    out = cp.zeros(out_shape, dtype=inp.dtype)
    _ff.restriction(
        out,
        inp,
        as_spline(order),
        as_bound(bound),
        shift,
        scale,
        ndim,
        current_stream_ptr(),
    )
    return out


def spline_coeff(
    inp: Any, order: int | str = 3, bound: int | str = "dct2"
) -> Any:
    """Spline coefficients of a signal, along the last axis.

    Converts samples into the coefficients of the interpolating spline of order
    `order`, so that evaluating that spline at the sample positions gives the
    samples back. For orders 0 and 1 the coefficients are the samples. Only the
    last axis is filtered.

    For orders above 1 only the boundary conditions `dct1`, `dct2`, `dft` and
    `replicate` are implemented; any other is rejected. The coefficients are
    what [`pull`][fastfields.cupy.pull], [`push`][fastfields.cupy.push] and
    [`grad`][fastfields.cupy.grad] expect as their volume.

    [`spline_coeff_`][fastfields.cupy.spline_coeff_] is the in-place version.

    Parameters
    ----------
    inp : cupy.ndarray
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
    cupy.ndarray
        The coefficients, a new array the shape of `inp`.

    Raises
    ------
    TypeError
        If `inp` is not a float32 or float64 CuPy array.
    """
    out = as_gpu_array(inp, name="inp").copy()
    _ff.spline_coeff(
        out, as_spline(order), as_bound(bound), current_stream_ptr()
    )
    return out


def spline_coeff_(
    inp_out: Any, order: int | str = 3, bound: int | str = "dct2"
) -> Any:
    """Spline coefficients of a signal along the last axis, in place.

    The in-place version of [`spline_coeff`][fastfields.cupy.spline_coeff]: the
    samples in `inp_out` are replaced by the coefficients. For orders above 1
    only the boundary conditions `dct1`, `dct2`, `dft` and `replicate` are
    implemented; any other is rejected.

    Parameters
    ----------
    inp_out : cupy.ndarray
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
    cupy.ndarray
        `inp_out` itself, now holding the coefficients.

    Raises
    ------
    TypeError
        If `inp_out` is not a float32 or float64 CuPy array.
    """
    inp_out = require_gpu_writethrough(inp_out, name="inp_out")
    _ff.spline_coeff(
        inp_out, as_spline(order), as_bound(bound), current_stream_ptr()
    )
    return inp_out
