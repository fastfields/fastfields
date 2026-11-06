"""Distance-transform wrappers for NumPy.

Euclidean and L1 distance transforms along the last axis, and point-to-spline
and point-to-mesh distances.
"""

from __future__ import annotations

import fastfields.dlpack as _ff

import numpy as np

from ._util import (
    _as_bound,
    _as_float_array,
    _as_spline,
    _broadcast_batch,
    _validate_inplace,
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
# distance transforms                                                         #
# --------------------------------------------------------------------------- #
#
# Trailing-underscore wrappers operate in place / through the caller's buffer
# and return it; this matches the ``_``-suffix convention already used by
# sym_addmatvec_/sym_submatvec_ in this same package and by every in-place op
# on the cupy backend (fastfields-cupy's ``dt_euclidean_``/``dt_l1_``). Earlier
# revisions spelled this as an ``inplace=`` keyword instead of a dedicated
# function; that was an accidental (not load-bearing) divergence from the
# rest of the package and from cupy -- see fastfields#4.


def dt_euclidean(inp: np.ndarray, voxel_spacing: float = 1.0) -> np.ndarray:
    """Squared Euclidean distance transform along the last axis.

    The input marks the features with `0` and everything else with `+inf`. Each
    position of the result holds the squared distance to the nearest feature in
    the same row along the last axis, in units set by `voxel_spacing`. Only the
    last axis is transformed.

    [`dt_euclidean_`][fastfields.numpy.dt_euclidean_] is the in-place version,
    which overwrites its input.

    Parameters
    ----------
    inp : numpy.ndarray
        Features are `0`, background is `+inf`. Arrays that are not float32
        or float64 are converted to float64.
    voxel_spacing : float, default=1.0
        Physical distance between neighbouring samples along the last axis.

    Returns
    -------
    numpy.ndarray
        The distance transform, a new array the shape of `inp`.
    """
    out = _as_float_array(inp, "inp", copy=True)
    _ff.dt_euclidean(out, float(voxel_spacing))
    return out


def dt_euclidean_(
    inp_out: np.ndarray, voxel_spacing: float = 1.0
) -> np.ndarray:
    """Squared Euclidean distance transform along the last axis, in place.

    The input marks the features with `0` and everything else with `+inf`. Each
    position of the result holds the squared distance to the nearest feature in
    the same row along the last axis, in units set by `voxel_spacing`. Only the
    last axis is transformed.

    [`dt_euclidean`][fastfields.numpy.dt_euclidean] is the version that returns
    a new array and leaves its input untouched.

    Parameters
    ----------
    inp_out : numpy.ndarray
        Features are `0`, background is `+inf`. Must be float32 or float64.
        Overwritten with the result.
    voxel_spacing : float, default=1.0
        Physical distance between neighbouring samples along the last axis.

    Returns
    -------
    numpy.ndarray
        `inp_out` itself, now holding the distance transform.

    Raises
    ------
    TypeError
        If `inp_out` is not a float32 or float64 NumPy array.
    """
    _validate_inplace(inp_out, "inp_out")
    _ff.dt_euclidean(inp_out, float(voxel_spacing))
    return inp_out


def dt_l1(inp: np.ndarray, voxel_spacing: float = 1.0) -> np.ndarray:
    """L1 distance transform along the last axis.

    The input marks the features with `0` and everything else with `+inf`. Each
    position of the result holds the distance to the nearest feature in the
    same row along the last axis, in units set by `voxel_spacing`. Only the
    last axis is transformed.

    [`dt_l1_`][fastfields.numpy.dt_l1_] is the in-place version, which
    overwrites its input.

    Parameters
    ----------
    inp : numpy.ndarray
        Features are `0`, background is `+inf`. Arrays that are not float32
        or float64 are converted to float64.
    voxel_spacing : float, default=1.0
        Physical distance between neighbouring samples along the last axis.

    Returns
    -------
    numpy.ndarray
        The distance transform, a new array the shape of `inp`.
    """
    out = _as_float_array(inp, "inp", copy=True)
    _ff.dt_l1(out, float(voxel_spacing))
    return out


def dt_l1_(inp_out: np.ndarray, voxel_spacing: float = 1.0) -> np.ndarray:
    """L1 distance transform along the last axis, in place.

    The input marks the features with `0` and everything else with `+inf`. Each
    position of the result holds the distance to the nearest feature in the
    same row along the last axis, in units set by `voxel_spacing`. Only the
    last axis is transformed.

    [`dt_l1`][fastfields.numpy.dt_l1] is the version that returns a new array
    and leaves its input untouched.

    Parameters
    ----------
    inp_out : numpy.ndarray
        Features are `0`, background is `+inf`. Must be float32 or float64.
        Overwritten with the result.
    voxel_spacing : float, default=1.0
        Physical distance between neighbouring samples along the last axis.

    Returns
    -------
    numpy.ndarray
        `inp_out` itself, now holding the distance transform.

    Raises
    ------
    TypeError
        If `inp_out` is not a float32 or float64 NumPy array.
    """
    _validate_inplace(inp_out, "inp_out")
    _ff.dt_l1(inp_out, float(voxel_spacing))
    return inp_out


# --------------------------------------------------------------------------- #
# point-to-spline / point-to-mesh distance                                    #
# --------------------------------------------------------------------------- #
#
# NOTE: the underlying fastfields-cpu-lib shape checks for these ops are
# inconsistent with the kernels (see this package's report / README notes),
# so they are exposed as thin pass-throughs following the documented
# distance.h contract but are not otherwise validated here.


def dt_spline_table(
    loc: np.ndarray,
    coeff: np.ndarray,
    times: np.ndarray,
    *,
    order: int | str = 3,
    bound: int | str = "dct2",
) -> tuple[np.ndarray, np.ndarray]:
    """Squared distance from points to a spline, over candidate parameters.

    The curve is a spline with `N` coefficients in `D` dimensions. For each
    query point the closest point of the curve is searched for. The result is
    the pair ``(dist, time)``: the squared distance and the curve parameter at
    which it is attained. Batch axes of all inputs are broadcast against each
    other.

    The closest point is the best of the candidate parameters in `times`, so
    the accuracy is limited by how finely they sample the curve.
    [`dt_spline_brent`][fastfields.numpy.dt_spline_brent] and
    [`dt_spline_gaussnewton`][fastfields.numpy.dt_spline_gaussnewton] refine
    the parameter iteratively instead.

    Parameters
    ----------
    loc : numpy.ndarray
        Query points, shape `(*B, D)`.
    coeff : numpy.ndarray
        Spline coefficients of the curve, shape `(*B, N, D)`: `N`
        coefficients, each a `D`-dimensional vector.
    times : numpy.ndarray
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
    dist : numpy.ndarray
        Squared distance from each query point to the curve, shape `(*B,)`.
    time : numpy.ndarray
        Curve parameter of the closest point found, shape `(*B,)`.
    """
    loc = _as_float_array(loc, "loc")
    coeff = _as_float_array(coeff, "coeff", dtype=loc.dtype)
    times = _as_float_array(times, "times", dtype=loc.dtype)
    batch, (loc_b, coeff_b, times_b) = _broadcast_batch(
        [(loc, 1), (coeff, 2), (times, 1)]
    )
    dist = np.empty(batch, dtype=loc.dtype)
    time = np.empty(batch, dtype=loc.dtype)
    _ff.dt_spline_table(
        time,
        dist,
        loc_b,
        coeff_b,
        times_b,
        _as_spline(order),
        _as_bound(bound),
    )
    return dist, time


def dt_spline_brent(
    loc: np.ndarray,
    coeff: np.ndarray,
    *,
    max_iter: int = 64,
    tol: float = 1e-6,
    step: float = 0.1,
    order: int | str = 3,
    bound: int | str = "dct2",
) -> tuple[np.ndarray, np.ndarray]:
    """Squared distance from points to a spline, refined with Brent's method.

    The curve is a spline with `N` coefficients in `D` dimensions. For each
    query point the closest point of the curve is searched for. The result is
    the pair ``(dist, time)``: the squared distance and the curve parameter at
    which it is attained. Batch axes of all inputs are broadcast against each
    other.

    The curve parameter is refined by Brent's method, which needs a bracket
    around the optimum; `step` sets how wide the initial one is.
    [`dt_spline_table`][fastfields.numpy.dt_spline_table] and
    [`dt_spline_gaussnewton`][fastfields.numpy.dt_spline_gaussnewton] are the
    alternatives.

    Parameters
    ----------
    loc : numpy.ndarray
        Query points, shape `(*B, D)`.
    coeff : numpy.ndarray
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
    dist : numpy.ndarray
        Squared distance from each query point to the curve, shape `(*B,)`.
    time : numpy.ndarray
        Curve parameter of the closest point found, shape `(*B,)`.
    """
    loc = _as_float_array(loc, "loc")
    coeff = _as_float_array(coeff, "coeff", dtype=loc.dtype)
    batch, (loc_b, coeff_b) = _broadcast_batch([(loc, 1), (coeff, 2)])
    dist = np.empty(batch, dtype=loc.dtype)
    time = np.empty(batch, dtype=loc.dtype)
    _ff.dt_spline_brent(
        time,
        dist,
        loc_b,
        coeff_b,
        int(max_iter),
        float(tol),
        float(step),
        _as_spline(order),
        _as_bound(bound),
    )
    return dist, time


def dt_spline_gaussnewton(
    loc: np.ndarray,
    coeff: np.ndarray,
    *,
    max_iter: int = 64,
    tol: float = 1e-6,
    order: int | str = 3,
    bound: int | str = "dct2",
) -> tuple[np.ndarray, np.ndarray]:
    """Squared distance from points to a spline, refined with Gauss-Newton.

    The curve is a spline with `N` coefficients in `D` dimensions. For each
    query point the closest point of the curve is searched for. The result is
    the pair ``(dist, time)``: the squared distance and the curve parameter at
    which it is attained. Batch axes of all inputs are broadcast against each
    other.

    The curve parameter is refined by Gauss-Newton iterations.
    [`dt_spline_table`][fastfields.numpy.dt_spline_table] and
    [`dt_spline_brent`][fastfields.numpy.dt_spline_brent] are the alternatives.

    Parameters
    ----------
    loc : numpy.ndarray
        Query points, shape `(*B, D)`.
    coeff : numpy.ndarray
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
    dist : numpy.ndarray
        Squared distance from each query point to the curve, shape `(*B,)`.
    time : numpy.ndarray
        Curve parameter of the closest point found, shape `(*B,)`.
    """
    loc = _as_float_array(loc, "loc")
    coeff = _as_float_array(coeff, "coeff", dtype=loc.dtype)
    batch, (loc_b, coeff_b) = _broadcast_batch([(loc, 1), (coeff, 2)])
    dist = np.empty(batch, dtype=loc.dtype)
    time = np.empty(batch, dtype=loc.dtype)
    _ff.dt_spline_gaussnewton(
        time,
        dist,
        loc_b,
        coeff_b,
        int(max_iter),
        float(tol),
        _as_spline(order),
        _as_bound(bound),
    )
    return dist, time


def dt_mesh(
    loc: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    signed: bool = True,
    naive: bool = False,
    return_nearest: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Distance from points to a triangular mesh.

    A single mesh is queried by any number of points: only `loc` has batch
    axes, and the result has shape `loc.shape[:-1]`. `vertices` and `faces` are
    not broadcast against that batch and must be exactly two-dimensional.

    The distance is the plain Euclidean point-to-triangle distance, not
    squared, unlike the point-to-spline distances. With `signed=True` it is
    negative inside the surface.

    Parameters
    ----------
    loc : numpy.ndarray
        Query points, shape `(*B, D)`.
    vertices : numpy.ndarray
        Mesh vertices, shape `(V, D)`.
    faces : numpy.ndarray
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
    dist : numpy.ndarray
        Distance from each query point to the mesh, shape `(*B,)`.
    nearest : numpy.ndarray
        Index of the vertex nearest to each query point, shape `(*B,)`,
        int64. Returned only if `return_nearest` is true.

    Raises
    ------
    ValueError
        If `vertices` or `faces` is not two-dimensional.
    """
    loc = _as_float_array(loc, "loc")
    vertices = _as_float_array(vertices, "vertices", dtype=loc.dtype)
    # faces is an integer index array; keep native strides (stride-aware
    # kernel) but ensure it is writable so numpy can export it via DLPack.
    faces = np.asarray(faces, dtype=np.int64)
    if not faces.flags.writeable:
        faces = faces.copy()
    if vertices.ndim != 2:
        raise ValueError(
            "dt_mesh: vertices must be a 2D (N, D) array describing a single "
            f"mesh, got shape {vertices.shape}"
        )
    if faces.ndim != 2:
        raise ValueError(
            "dt_mesh: faces must be a 2D (M, D) array describing a single "
            f"mesh, got shape {faces.shape}"
        )
    batch = loc.shape[:-1]
    dist = np.empty(batch, dtype=loc.dtype)
    nearest = None
    if return_nearest:
        nearest = np.empty(batch, dtype=np.int64)
    _ff.dt_mesh(dist, nearest, loc, vertices, faces, bool(signed), bool(naive))
    if return_nearest:
        return dist, nearest
    return dist
