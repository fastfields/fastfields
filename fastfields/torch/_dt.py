"""Distance-transform wrappers (torch).

Image distance transforms (``dt_euclidean``, ``dt_l1``), point-to-mesh
distance (``dt_mesh``) and point-to-spline distance
(``dt_spline_{table,brent,gaussnewton}``).

These operators are **not differentiable** -- the distance transform / point-
to-mesh distance have no meaningful gradient. They are still exposed on torch
(for API parity with ``fastfields.numpy``/``fastfields.cupy``): forward always
runs, including when an input requires grad, so the output can sit inside a
larger graph; only calling ``.backward()`` through that output raises a clear
``RuntimeError`` (see `fastfields.torch._util.raise_not_differentiable`).
This mirrors how ``jitfields`` guards these ops, but the guard now lives in
``backward`` instead of at call time -- see ``API_CONTRACT.md``, "In-place
policy", and fastfields#4.

The underlying bindings mutate their input in place. The out-of-place
functions (``dt_euclidean``, ``dt_l1``) clone first; the in-place variants
(``dt_euclidean_``, ``dt_l1_``) write through the caller's tensor, exactly
like the numpy/cupy ``_``-suffixed forms. On CUDA tensors the current stream
is forwarded to the binding (see `fastfields.torch._util`).
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import fastfields.dlpack as _fb

import torch
from fastfields.helpers import as_bound, as_spline
from torch import Tensor

from ._util import (
    check_dtype,
    check_inplace_leaf,
    raise_not_differentiable,
    stream_ptr,
)

__all__ = [
    "dt_euclidean",
    "dt_euclidean_",
    "dt_l1",
    "dt_l1_",
    "dt_mesh",
    "dt_spline_table",
    "dt_spline_brent",
    "dt_spline_gaussnewton",
]


class _DtEuclidean(torch.autograd.Function):
    """``out = dt_euclidean(inp)`` (out-of-place, ``inp`` left untouched)."""

    @staticmethod
    def forward(ctx, inp, voxel_spacing):
        out = inp.clone()
        _fb.dt_euclidean(out, voxel_spacing, stream=stream_ptr(out))
        return out

    @staticmethod
    def backward(ctx, grad):
        raise_not_differentiable(
            "dt_euclidean",
            "the distance transform has no meaningful gradient.",
        )


class _DtEuclidean_(torch.autograd.Function):
    """``inp <- dt_euclidean(inp)`` in place; returns ``inp``."""

    @staticmethod
    def forward(ctx, inp, voxel_spacing):
        _fb.dt_euclidean(inp, voxel_spacing, stream=stream_ptr(inp))
        ctx.mark_dirty(inp)
        return inp

    @staticmethod
    def backward(ctx, grad):
        raise_not_differentiable(
            "dt_euclidean_",
            "the distance transform has no meaningful gradient.",
        )


def dt_euclidean(inp: Tensor, voxel_spacing: float = 1.0) -> Tensor:
    """Squared Euclidean distance transform along the last axis.

    The input marks the features with `0` and everything else with `+inf`. Each
    position of the result holds the squared distance to the nearest feature in
    the same row along the last axis, in units set by `voxel_spacing`. Only the
    last axis is transformed.

    [`dt_euclidean_`][fastfields.torch.dt_euclidean_] is the in-place version,
    which overwrites its input.

    Not differentiable: the forward pass runs normally, even when an input
    requires grad, but calling `.backward()` through the result raises
    `RuntimeError`.

    Parameters
    ----------
    inp : torch.Tensor
        Features are `0`, background is `+inf`.
    voxel_spacing : float, default=1.0
        Physical distance between neighbouring samples along the last axis.

    Returns
    -------
    torch.Tensor
        The distance transform, a new tensor the shape of `inp`.

    Raises
    ------
    TypeError
        If the tensor is not float32 or float64.
    RuntimeError
        If `.backward()` is called through the result.
    """
    check_dtype(inp)
    return _DtEuclidean.apply(inp, voxel_spacing)


def dt_euclidean_(inp_out: Tensor, voxel_spacing: float = 1.0) -> Tensor:
    """Squared Euclidean distance transform along the last axis, in place.

    The input marks the features with `0` and everything else with `+inf`. Each
    position of the result holds the squared distance to the nearest feature in
    the same row along the last axis, in units set by `voxel_spacing`. Only the
    last axis is transformed.

    [`dt_euclidean`][fastfields.torch.dt_euclidean] is the version that returns
    a new tensor and leaves its input untouched.

    Not differentiable: the forward pass runs normally, even when an input
    requires grad, but calling `.backward()` through the result raises
    `RuntimeError`.

    Parameters
    ----------
    inp_out : torch.Tensor
        Features are `0`, background is `+inf`. Must be float32 or float64.
        Overwritten with the result.
    voxel_spacing : float, default=1.0
        Physical distance between neighbouring samples along the last axis.

    Returns
    -------
    torch.Tensor
        `inp_out` itself, now holding the distance transform.

    Raises
    ------
    TypeError
        If the tensor is not float32 or float64.
    RuntimeError
        If `.backward()` is called through the result.
    RuntimeError
        If `inp_out` is a leaf tensor that requires grad. This is checked
        before anything is written.
    """
    check_dtype(inp_out)
    check_inplace_leaf("dt_euclidean_", inp_out)
    return _DtEuclidean_.apply(inp_out, voxel_spacing)


class _DtL1(torch.autograd.Function):
    """``out = dt_l1(inp)`` (out-of-place, ``inp`` left untouched)."""

    @staticmethod
    def forward(ctx, inp, voxel_spacing):
        out = inp.clone()
        _fb.dt_l1(out, voxel_spacing, stream=stream_ptr(out))
        return out

    @staticmethod
    def backward(ctx, grad):
        raise_not_differentiable(
            "dt_l1", "the distance transform has no meaningful gradient."
        )


class _DtL1_(torch.autograd.Function):
    """``inp <- dt_l1(inp)`` in place; returns ``inp``."""

    @staticmethod
    def forward(ctx, inp, voxel_spacing):
        _fb.dt_l1(inp, voxel_spacing, stream=stream_ptr(inp))
        ctx.mark_dirty(inp)
        return inp

    @staticmethod
    def backward(ctx, grad):
        raise_not_differentiable(
            "dt_l1_", "the distance transform has no meaningful gradient."
        )


def dt_l1(inp: Tensor, voxel_spacing: float = 1.0) -> Tensor:
    """L1 distance transform along the last axis.

    The input marks the features with `0` and everything else with `+inf`. Each
    position of the result holds the distance to the nearest feature in the
    same row along the last axis, in units set by `voxel_spacing`. Only the
    last axis is transformed.

    [`dt_l1_`][fastfields.torch.dt_l1_] is the in-place version, which
    overwrites its input.

    Not differentiable: the forward pass runs normally, even when an input
    requires grad, but calling `.backward()` through the result raises
    `RuntimeError`.

    Parameters
    ----------
    inp : torch.Tensor
        Features are `0`, background is `+inf`.
    voxel_spacing : float, default=1.0
        Physical distance between neighbouring samples along the last axis.

    Returns
    -------
    torch.Tensor
        The distance transform, a new tensor the shape of `inp`.

    Raises
    ------
    TypeError
        If the tensor is not float32 or float64.
    RuntimeError
        If `.backward()` is called through the result.
    """
    check_dtype(inp)
    return _DtL1.apply(inp, voxel_spacing)


def dt_l1_(inp_out: Tensor, voxel_spacing: float = 1.0) -> Tensor:
    """L1 distance transform along the last axis, in place.

    The input marks the features with `0` and everything else with `+inf`. Each
    position of the result holds the distance to the nearest feature in the
    same row along the last axis, in units set by `voxel_spacing`. Only the
    last axis is transformed.

    [`dt_l1`][fastfields.torch.dt_l1] is the version that returns a new tensor
    and leaves its input untouched.

    Not differentiable: the forward pass runs normally, even when an input
    requires grad, but calling `.backward()` through the result raises
    `RuntimeError`.

    Parameters
    ----------
    inp_out : torch.Tensor
        Features are `0`, background is `+inf`. Must be float32 or float64.
        Overwritten with the result.
    voxel_spacing : float, default=1.0
        Physical distance between neighbouring samples along the last axis.

    Returns
    -------
    torch.Tensor
        `inp_out` itself, now holding the distance transform.

    Raises
    ------
    TypeError
        If the tensor is not float32 or float64.
    RuntimeError
        If `.backward()` is called through the result.
    RuntimeError
        If `inp_out` is a leaf tensor that requires grad. This is checked
        before anything is written.
    """
    check_dtype(inp_out)
    check_inplace_leaf("dt_l1_", inp_out)
    return _DtL1_.apply(inp_out, voxel_spacing)


class _DtMesh(torch.autograd.Function):
    """Point-to-mesh (squared) distance; ``loc``/``vertices`` left untouched.

    No in-place variant exists on any backend (the output shape/target
    differs from every input, so there is no natural buffer to mutate).
    """

    @staticmethod
    def forward(ctx, loc, vertices, faces, signed, naive, return_nearest):
        # A single mesh is queried by an arbitrarily batched point set: only
        # `loc` carries batch dims, and `vertices`/`faces` stay unbatched
        # (the binding requires exactly (N, D) / (M, D)). Outputs are
        # contiguous real buffers shaped like loc.shape[:-1].
        batch = loc.shape[:-1]
        dist = loc.new_empty(batch)
        nearest = None
        if return_nearest:
            nearest = torch.empty(batch, dtype=torch.int64, device=loc.device)
        _fb.dt_mesh(
            dist,
            nearest,
            loc,
            vertices,
            faces,
            signed,
            naive,
            stream=stream_ptr(dist),
        )
        if return_nearest:
            return dist, nearest
        return dist

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise_not_differentiable(
            "dt_mesh", "point-to-mesh distance has no meaningful gradient."
        )


def dt_mesh(
    loc: Tensor,
    vertices: Tensor,
    faces: Tensor,
    signed: bool = True,
    naive: bool = False,
    return_nearest: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Distance from points to a triangular mesh.

    A single mesh is queried by any number of points: only `loc` has batch
    axes, and the result has shape `loc.shape[:-1]`. `vertices` and `faces` are
    not broadcast against that batch and must be exactly two-dimensional.

    The distance is the plain Euclidean point-to-triangle distance, not
    squared, unlike the point-to-spline distances. With `signed=True` it is
    negative inside the surface.

    Not differentiable: the forward pass runs normally, even when an input
    requires grad, but calling `.backward()` through the result raises
    `RuntimeError`.

    Parameters
    ----------
    loc : torch.Tensor
        Query points, shape `(*B, D)`.
    vertices : torch.Tensor
        Mesh vertices, shape `(V, D)`.
    faces : torch.Tensor
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
    dist : torch.Tensor
        Distance from each query point to the mesh, shape `(*B,)`.
    nearest : torch.Tensor
        Index of the vertex nearest to each query point, shape `(*B,)`,
        int64. Returned only if `return_nearest` is true.

    Raises
    ------
    TypeError
        If `loc` or `vertices` is not float32 or float64.
    ValueError
        If `vertices` or `faces` is not two-dimensional.
    RuntimeError
        If `.backward()` is called through `dist`.
    """
    check_dtype(loc, vertices)
    # faces holds integer vertex indices: the binding reads them at int64
    # width, so an int32 (or other) faces array would be misread. Normalize to
    # int64 before the binding (mirrors the numpy wrapper).
    if faces.dtype != torch.int64:
        faces = faces.to(torch.int64)
    if vertices.ndim != 2:
        raise ValueError(
            "dt_mesh: vertices must be a 2D (V, D) tensor describing a "
            f"single mesh, got shape {tuple(vertices.shape)}"
        )
    if faces.ndim != 2:
        raise ValueError(
            "dt_mesh: faces must be a 2D (F, D) tensor describing a single "
            f"mesh, got shape {tuple(faces.shape)}"
        )
    return _DtMesh.apply(loc, vertices, faces, signed, naive, return_nearest)


# --------------------------------------------------------------------------- #
# Point-to-spline distance                                                    #
# --------------------------------------------------------------------------- #
#
# Three strategies for the same problem -- find the time `t` minimising
# ||loc - spline(t)||^2 -- and the same contract: `loc` core dim (D,), `coeff`
# core dim (N, D), both outputs shaped like the broadcast batch. Like the
# other distance ops they are not differentiable (the result is the *argmin*
# of an inner optimisation, which the library does not differentiate), so
# forward always runs and only `.backward()` through an output raises.
#
# `dt_spline_brent` and `dt_spline_gaussnewton` *refine* the time/distance
# buffers they are handed, so these wrappers seed them: `time` with the
# midpoint of the spline's parameter range and `dist` with +inf, which is the
# cold start. Seed them yourself with a coarse `dt_spline_table` pass (and
# call the binding directly) if you have a better starting point.


def _spline_operands(loc: Tensor, coeff: Tensor, *rest: Tensor):
    """Validate and broadcast a (loc, coeff, *rest) spline-distance call."""
    check_dtype(loc, coeff, *rest)
    if loc.ndim < 1:
        raise ValueError("dt_spline_*: loc must have at least one axis (D,)")
    if coeff.ndim < 2:
        raise ValueError(
            "dt_spline_*: coeff must have at least two axes (N, D)"
        )
    if coeff.shape[-1] != loc.shape[-1]:
        raise ValueError(
            f"dt_spline_*: loc has D={loc.shape[-1]} coordinates but coeff "
            f"has D={coeff.shape[-1]}"
        )
    shapes = [loc.shape[:-1], coeff.shape[:-2]]
    shapes += [t.shape[:-1] for t in rest]
    batch = torch.broadcast_shapes(*shapes)
    out = [
        loc.expand(*batch, loc.shape[-1]),
        coeff.expand(*batch, *coeff.shape[-2:]),
    ]
    out += [t.expand(*batch, t.shape[-1]) for t in rest]
    return batch, out


def _spline_outputs(loc: Tensor, batch: Sequence[int], seed: bool):
    """Allocate the (time, dist) output buffers for a spline-distance call."""
    time = loc.new_empty(tuple(batch))
    dist = loc.new_empty(tuple(batch))
    if seed:
        # Brent / Gauss-Newton refine whatever is in the buffers.
        time.fill_(0.0)
        dist.fill_(float("inf"))
    return time, dist


class _DtSpline(torch.autograd.Function):
    """Any of the three point-to-spline distance ops; not differentiable.

    ``loc`` / ``coeff`` / ``times`` are passed as real ``Function`` inputs,
    not captured in a closure, so that autograd records the edge: if one of
    them requires grad the outputs do too, and a ``.backward()`` through them
    reaches `backward` and gets the op-specific message rather than
    torch's generic "does not require grad". That is the contract the other
    non-differentiable ops in this module follow.
    """

    @staticmethod
    def forward(ctx, kind, time, dist, loc, coeff, times, params):
        spline, bound, max_iter, tol, step = params
        if kind == "dt_spline_table":
            _fb.dt_spline_table(
                time,
                dist,
                loc,
                coeff,
                times,
                spline=spline,
                bound=bound,
                stream=stream_ptr(dist),
            )
        elif kind == "dt_spline_brent":
            _fb.dt_spline_brent(
                time,
                dist,
                loc,
                coeff,
                max_iter,
                tol,
                step,
                spline=spline,
                bound=bound,
                stream=stream_ptr(dist),
            )
        else:
            _fb.dt_spline_gaussnewton(
                time,
                dist,
                loc,
                coeff,
                max_iter,
                tol,
                spline=spline,
                bound=bound,
                stream=stream_ptr(dist),
            )
        ctx.kind = kind
        return dist, time

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise_not_differentiable(
            ctx.kind,
            "point-to-spline distance is the value at an argmin found by an "
            "inner optimisation, which the library does not differentiate.",
        )


def dt_spline_table(
    loc: Tensor,
    coeff: Tensor,
    times: Tensor,
    *,
    order: int | str = 3,
    bound: int | str = "dct2",
) -> Tuple[Tensor, Tensor]:
    """Squared distance from points to a spline, over candidate parameters.

    The curve is a spline with `N` coefficients in `D` dimensions. For each
    query point the closest point of the curve is searched for. The result is
    the pair ``(dist, time)``: the squared distance and the curve parameter at
    which it is attained. Batch axes of all inputs are broadcast against each
    other.

    The closest point is the best of the candidate parameters in `times`, so
    the accuracy is limited by how finely they sample the curve.
    [`dt_spline_brent`][fastfields.torch.dt_spline_brent] and
    [`dt_spline_gaussnewton`][fastfields.torch.dt_spline_gaussnewton] refine
    the parameter iteratively instead.

    Not differentiable: the forward pass runs normally, even when an input
    requires grad, but calling `.backward()` through the result raises
    `RuntimeError`. The result comes from an inner optimisation that the
    library does not differentiate.

    Parameters
    ----------
    loc : torch.Tensor
        Query points, shape `(*B, D)`.
    coeff : torch.Tensor
        Spline coefficients of the curve, shape `(*B, N, D)`: `N`
        coefficients, each a `D`-dimensional vector.
    times : torch.Tensor
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
    dist : torch.Tensor
        Squared distance from each query point to the curve, shape `(*B,)`.
    time : torch.Tensor
        Curve parameter of the closest point found, shape `(*B,)`.
    """
    batch, (loc, coeff, times) = _spline_operands(loc, coeff, times)
    time, dist = _spline_outputs(loc, batch, seed=False)
    return _DtSpline.apply(
        "dt_spline_table",
        time,
        dist,
        loc,
        coeff,
        times,
        (as_spline(order), as_bound(bound), 0, 0.0, 0.0),
    )


def dt_spline_brent(
    loc: Tensor,
    coeff: Tensor,
    *,
    max_iter: int = 64,
    tol: float = 1e-6,
    step: float = 0.1,
    order: int | str = 3,
    bound: int | str = "dct2",
) -> Tuple[Tensor, Tensor]:
    """Squared distance from points to a spline, refined with Brent's method.

    The curve is a spline with `N` coefficients in `D` dimensions. For each
    query point the closest point of the curve is searched for. The result is
    the pair ``(dist, time)``: the squared distance and the curve parameter at
    which it is attained. Batch axes of all inputs are broadcast against each
    other.

    The curve parameter is refined by Brent's method, which needs a bracket
    around the optimum; `step` sets how wide the initial one is.
    [`dt_spline_table`][fastfields.torch.dt_spline_table] and
    [`dt_spline_gaussnewton`][fastfields.torch.dt_spline_gaussnewton] are the
    alternatives.

    Not differentiable: the forward pass runs normally, even when an input
    requires grad, but calling `.backward()` through the result raises
    `RuntimeError`. The result comes from an inner optimisation that the
    library does not differentiate.

    Parameters
    ----------
    loc : torch.Tensor
        Query points, shape `(*B, D)`.
    coeff : torch.Tensor
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
    dist : torch.Tensor
        Squared distance from each query point to the curve, shape `(*B,)`.
    time : torch.Tensor
        Curve parameter of the closest point found, shape `(*B,)`.
    """
    batch, (loc, coeff) = _spline_operands(loc, coeff)
    time, dist = _spline_outputs(loc, batch, seed=True)
    return _DtSpline.apply(
        "dt_spline_brent",
        time,
        dist,
        loc,
        coeff,
        None,
        (
            as_spline(order),
            as_bound(bound),
            int(max_iter),
            float(tol),
            float(step),
        ),
    )


def dt_spline_gaussnewton(
    loc: Tensor,
    coeff: Tensor,
    *,
    max_iter: int = 64,
    tol: float = 1e-6,
    order: int | str = 3,
    bound: int | str = "dct2",
) -> Tuple[Tensor, Tensor]:
    """Squared distance from points to a spline, refined with Gauss-Newton.

    The curve is a spline with `N` coefficients in `D` dimensions. For each
    query point the closest point of the curve is searched for. The result is
    the pair ``(dist, time)``: the squared distance and the curve parameter at
    which it is attained. Batch axes of all inputs are broadcast against each
    other.

    The curve parameter is refined by Gauss-Newton iterations.
    [`dt_spline_table`][fastfields.torch.dt_spline_table] and
    [`dt_spline_brent`][fastfields.torch.dt_spline_brent] are the alternatives.

    Not differentiable: the forward pass runs normally, even when an input
    requires grad, but calling `.backward()` through the result raises
    `RuntimeError`. The result comes from an inner optimisation that the
    library does not differentiate.

    Parameters
    ----------
    loc : torch.Tensor
        Query points, shape `(*B, D)`.
    coeff : torch.Tensor
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
    dist : torch.Tensor
        Squared distance from each query point to the curve, shape `(*B,)`.
    time : torch.Tensor
        Curve parameter of the closest point found, shape `(*B,)`.
    """
    batch, (loc, coeff) = _spline_operands(loc, coeff)
    time, dist = _spline_outputs(loc, batch, seed=True)
    return _DtSpline.apply(
        "dt_spline_gaussnewton",
        time,
        dist,
        loc,
        coeff,
        None,
        (as_spline(order), as_bound(bound), int(max_iter), float(tol), 0.0),
    )
