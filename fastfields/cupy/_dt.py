"""Distance-transform wrappers for CuPy.

Euclidean and L1 distance transforms along the last axis, and point-to-spline
and point-to-mesh distances.
"""

from __future__ import annotations

from typing import Any

import fastfields.dlpack as _ff

from fastfields.helpers import as_bound, as_spline

from ._util import (
    as_gpu_array,
    broadcast_batch,
    cupy,
    current_stream_ptr,
    require_gpu_writethrough,
)

__all__ = [
    "dt_euclidean",
    "dt_euclidean_",
    "dt_l1",
    "dt_l1_",
    "dt_spline_table",
    "dt_spline_brent",
    "dt_spline_gaussnewton",
    "dt_mesh",
]


# --------------------------------------------------------------------------- #
# Euclidean / L1 distance transform (in-place along the last axis)            #
# --------------------------------------------------------------------------- #
def dt_euclidean(inp: Any, voxel_spacing: float = 1.0) -> Any:
    """Squared Euclidean distance transform along the last axis.

    The input marks the features with `0` and everything else with `+inf`. Each
    position of the result holds the squared distance to the nearest feature in
    the same row along the last axis, in units set by `voxel_spacing`. Only the
    last axis is transformed.

    [`dt_euclidean_`][fastfields.cupy.dt_euclidean_] is the in-place version,
    which overwrites its input.

    Parameters
    ----------
    inp : cupy.ndarray
        Features are `0`, background is `+inf`.
    voxel_spacing : float, default=1.0
        Physical distance between neighbouring samples along the last axis.

    Returns
    -------
    cupy.ndarray
        The distance transform, a new array the shape of `inp`.

    Raises
    ------
    TypeError
        If the argument is not a float32 or float64 CuPy array.
    """
    out = as_gpu_array(inp, name="inp").copy()
    _ff.dt_euclidean(out, voxel_spacing, current_stream_ptr())
    return out


def dt_euclidean_(inp_out: Any, voxel_spacing: float = 1.0) -> Any:
    """Squared Euclidean distance transform along the last axis, in place.

    The input marks the features with `0` and everything else with `+inf`. Each
    position of the result holds the squared distance to the nearest feature in
    the same row along the last axis, in units set by `voxel_spacing`. Only the
    last axis is transformed.

    [`dt_euclidean`][fastfields.cupy.dt_euclidean] is the version that returns
    a new array and leaves its input untouched.

    Parameters
    ----------
    inp_out : cupy.ndarray
        Features are `0`, background is `+inf`. Must be float32 or float64.
        Overwritten with the result.
    voxel_spacing : float, default=1.0
        Physical distance between neighbouring samples along the last axis.

    Returns
    -------
    cupy.ndarray
        `inp_out` itself, now holding the distance transform.

    Raises
    ------
    TypeError
        If the argument is not a float32 or float64 CuPy array.
    """
    inp_out = require_gpu_writethrough(inp_out, name="inp_out")
    _ff.dt_euclidean(inp_out, voxel_spacing, current_stream_ptr())
    return inp_out


def dt_l1(inp: Any, voxel_spacing: float = 1.0) -> Any:
    """L1 distance transform along the last axis.

    The input marks the features with `0` and everything else with `+inf`. Each
    position of the result holds the distance to the nearest feature in the
    same row along the last axis, in units set by `voxel_spacing`. Only the
    last axis is transformed.

    [`dt_l1_`][fastfields.cupy.dt_l1_] is the in-place version, which
    overwrites its input.

    Parameters
    ----------
    inp : cupy.ndarray
        Features are `0`, background is `+inf`.
    voxel_spacing : float, default=1.0
        Physical distance between neighbouring samples along the last axis.

    Returns
    -------
    cupy.ndarray
        The distance transform, a new array the shape of `inp`.

    Raises
    ------
    TypeError
        If the argument is not a float32 or float64 CuPy array.
    """
    out = as_gpu_array(inp, name="inp").copy()
    _ff.dt_l1(out, voxel_spacing, current_stream_ptr())
    return out


def dt_l1_(inp_out: Any, voxel_spacing: float = 1.0) -> Any:
    """L1 distance transform along the last axis, in place.

    The input marks the features with `0` and everything else with `+inf`. Each
    position of the result holds the distance to the nearest feature in the
    same row along the last axis, in units set by `voxel_spacing`. Only the
    last axis is transformed.

    [`dt_l1`][fastfields.cupy.dt_l1] is the version that returns a new array
    and leaves its input untouched.

    Parameters
    ----------
    inp_out : cupy.ndarray
        Features are `0`, background is `+inf`. Must be float32 or float64.
        Overwritten with the result.
    voxel_spacing : float, default=1.0
        Physical distance between neighbouring samples along the last axis.

    Returns
    -------
    cupy.ndarray
        `inp_out` itself, now holding the distance transform.

    Raises
    ------
    TypeError
        If the argument is not a float32 or float64 CuPy array.
    """
    inp_out = require_gpu_writethrough(inp_out, name="inp_out")
    _ff.dt_l1(inp_out, voxel_spacing, current_stream_ptr())
    return inp_out


# --------------------------------------------------------------------------- #
# Point-to-spline distance                                                    #
# --------------------------------------------------------------------------- #
#
# The three wrappers below return ``(dist, time)``, matching
# ``fastfields.numpy`` and ``fastfields.torch`` (fastfields#38). Note that the
# raw binding underneath takes its two output buffers the other way round --
# ``_ff.dt_spline_*(time, dist, ...)``, the order the C++ hub declares -- so
# the argument order at the call sites is *not* a copy of the return order and
# must not be "tidied" to match it.
def dt_spline_table(
    loc: Any,
    coeff: Any,
    times: Any,
    *,
    order: int | str = 3,
    bound: int | str = "dct2",
) -> tuple[Any, Any]:
    """Squared distance from points to a spline, over candidate parameters.

    The curve is a spline with `N` coefficients in `D` dimensions. For each
    query point the closest point of the curve is searched for. The result is
    the pair ``(dist, time)``: the squared distance and the curve parameter at
    which it is attained. Batch axes of all inputs are broadcast against each
    other.

    The closest point is the best of the candidate parameters in `times`, so
    the accuracy is limited by how finely they sample the curve.
    [`dt_spline_brent`][fastfields.cupy.dt_spline_brent] and
    [`dt_spline_gaussnewton`][fastfields.cupy.dt_spline_gaussnewton] refine the
    parameter iteratively instead.

    Parameters
    ----------
    loc : cupy.ndarray
        Query points, shape `(*B, D)`.
    coeff : cupy.ndarray
        Spline coefficients of the curve, shape `(*B, N, D)`: `N`
        coefficients, each a `D`-dimensional vector.
    times : cupy.ndarray
        Candidate curve parameters to try, shape `(*B, K)`.
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
    dist : cupy.ndarray
        Squared distance from each query point to the curve, shape `(*B,)`.
    time : cupy.ndarray
        Curve parameter of the closest point found, shape `(*B,)`.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    cp = cupy()
    loc = as_gpu_array(loc, name="loc")
    coeff = as_gpu_array(coeff, name="coeff")
    times = as_gpu_array(times, name="times")
    batch, (loc_b, coeff_b, times_b) = broadcast_batch(
        [(loc, 1), (coeff, 2), (times, 1)]
    )
    time = cp.empty(batch, dtype=loc.dtype)
    dist = cp.empty(batch, dtype=loc.dtype)
    _ff.dt_spline_table(
        time,
        dist,
        loc_b,
        coeff_b,
        times_b,
        as_spline(order),
        as_bound(bound),
        current_stream_ptr(),
    )
    return dist, time


def dt_spline_brent(
    loc: Any,
    coeff: Any,
    *,
    max_iter: int = 64,
    tol: float = 1e-6,
    step: float = 0.1,
    order: int | str = 3,
    bound: int | str = "dct2",
) -> tuple[Any, Any]:
    """Squared distance from points to a spline, refined with Brent's method.

    The curve is a spline with `N` coefficients in `D` dimensions. For each
    query point the closest point of the curve is searched for. The result is
    the pair ``(dist, time)``: the squared distance and the curve parameter at
    which it is attained. Batch axes of all inputs are broadcast against each
    other.

    The curve parameter is refined by Brent's method, which needs a bracket
    around the optimum; `step` sets how wide the initial one is.
    [`dt_spline_table`][fastfields.cupy.dt_spline_table] and
    [`dt_spline_gaussnewton`][fastfields.cupy.dt_spline_gaussnewton] are the
    alternatives.

    Parameters
    ----------
    loc : cupy.ndarray
        Query points, shape `(*B, D)`.
    coeff : cupy.ndarray
        Spline coefficients of the curve, shape `(*B, N, D)`: `N`
        coefficients, each a `D`-dimensional vector.
    max_iter : int, default=64
        Maximum number of iterations.
    tol : float, default=1e-06
        Convergence tolerance.
    step : float, default=0.1
        Initial bracketing step; it must be wide enough to bracket the
        optimum.
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
    dist : cupy.ndarray
        Squared distance from each query point to the curve, shape `(*B,)`.
    time : cupy.ndarray
        Curve parameter of the closest point found, shape `(*B,)`.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    cp = cupy()
    loc = as_gpu_array(loc, name="loc")
    coeff = as_gpu_array(coeff, name="coeff")
    batch, (loc_b, coeff_b) = broadcast_batch([(loc, 1), (coeff, 2)])
    time = cp.empty(batch, dtype=loc.dtype)
    dist = cp.empty(batch, dtype=loc.dtype)
    _ff.dt_spline_brent(
        time,
        dist,
        loc_b,
        coeff_b,
        int(max_iter),
        float(tol),
        float(step),
        as_spline(order),
        as_bound(bound),
        current_stream_ptr(),
    )
    return dist, time


def dt_spline_gaussnewton(
    loc: Any,
    coeff: Any,
    *,
    max_iter: int = 64,
    tol: float = 1e-6,
    order: int | str = 3,
    bound: int | str = "dct2",
) -> tuple[Any, Any]:
    """Squared distance from points to a spline, refined with Gauss-Newton.

    The curve is a spline with `N` coefficients in `D` dimensions. For each
    query point the closest point of the curve is searched for. The result is
    the pair ``(dist, time)``: the squared distance and the curve parameter at
    which it is attained. Batch axes of all inputs are broadcast against each
    other.

    The curve parameter is refined by Gauss-Newton iterations.
    [`dt_spline_table`][fastfields.cupy.dt_spline_table] and
    [`dt_spline_brent`][fastfields.cupy.dt_spline_brent] are the alternatives.

    Parameters
    ----------
    loc : cupy.ndarray
        Query points, shape `(*B, D)`.
    coeff : cupy.ndarray
        Spline coefficients of the curve, shape `(*B, N, D)`: `N`
        coefficients, each a `D`-dimensional vector.
    max_iter : int, default=64
        Maximum number of iterations.
    tol : float, default=1e-06
        Convergence tolerance.
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
    dist : cupy.ndarray
        Squared distance from each query point to the curve, shape `(*B,)`.
    time : cupy.ndarray
        Curve parameter of the closest point found, shape `(*B,)`.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    cp = cupy()
    loc = as_gpu_array(loc, name="loc")
    coeff = as_gpu_array(coeff, name="coeff")
    batch, (loc_b, coeff_b) = broadcast_batch([(loc, 1), (coeff, 2)])
    time = cp.empty(batch, dtype=loc.dtype)
    dist = cp.empty(batch, dtype=loc.dtype)
    _ff.dt_spline_gaussnewton(
        time,
        dist,
        loc_b,
        coeff_b,
        int(max_iter),
        float(tol),
        as_spline(order),
        as_bound(bound),
        current_stream_ptr(),
    )
    return dist, time


# --------------------------------------------------------------------------- #
# Point-to-mesh distance                                                      #
# --------------------------------------------------------------------------- #
def dt_mesh(
    loc: Any,
    vertices: Any,
    faces: Any,
    signed: bool = True,
    naive: bool = False,
    return_nearest: bool = False,
) -> Any:
    """Distance from points to a triangular mesh.

    A single mesh is queried by any number of points: only `loc` has batch
    axes, and the result has shape `loc.shape[:-1]`. `vertices` and `faces` are
    not broadcast against that batch and must be exactly two-dimensional.

    The distance is the plain Euclidean point-to-triangle distance, not
    squared, unlike the point-to-spline distances. With `signed=True` it is
    negative inside the surface.

    Parameters
    ----------
    loc : cupy.ndarray
        Query points, shape `(*B, D)`.
    vertices : cupy.ndarray
        Mesh vertices, shape `(V, D)`.
    faces : cupy.ndarray
        Triangles as indices into `vertices`, shape `(F, D)`. Converted to
        int64.
    signed : bool, default=True
        Return signed distances, negative inside the surface.
    naive : bool, default=False
        Use the brute-force algorithm instead of the acceleration
        structure.
    return_nearest : bool, default=False
        Also return the index of the nearest mesh vertex for every query
        point.

    Returns
    -------
    dist : cupy.ndarray
        Distance from each query point to the mesh, shape `(*B,)`.
    nearest : cupy.ndarray
        Index of the vertex nearest to each query point, shape `(*B,)`,
        int64. Returned only if `return_nearest` is true.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    cp = cupy()
    loc = as_gpu_array(loc, name="loc")
    vertices = as_gpu_array(vertices, name="vertices")
    faces = cp.asarray(faces)  # native strides preserved (stride-aware kernel)
    if vertices.ndim != 2:
        raise ValueError(
            "dt_mesh: vertices must be a 2D (V, D) array describing a single "
            f"mesh, got shape {vertices.shape}"
        )
    if faces.ndim != 2:
        raise ValueError(
            "dt_mesh: faces must be a 2D (M, D) array describing a single "
            f"mesh, got shape {faces.shape}"
        )
    batch = loc.shape[:-1]
    dist = cp.empty(batch, dtype=loc.dtype)
    nearest = cp.empty(batch, dtype=cp.int64) if return_nearest else None
    _ff.dt_mesh(
        dist,
        nearest,
        loc,
        vertices,
        faces,
        signed,
        naive,
        current_stream_ptr(),
    )
    if return_nearest:
        return dist, nearest
    return dist
