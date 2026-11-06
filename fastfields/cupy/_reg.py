"""Spatial regularisers — cupy.

* **field** — multi-channel field ``(*batch, *spatial, C)``; per-channel
  ``absolute`` / ``membrane`` / ``bending`` (a scalar broadcasts to ``C``).
* **flow** — vector flow field; scalar penalties.

Signatures match the numpy/torch wrappers so ``fastfields.auto`` dispatches
consistently.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import fastfields.dlpack as _ff

from fastfields.helpers import as_bound

from ._util import (
    as_gpu_array,
    broadcast_batch,
    cupy,
    current_stream_ptr,
    require_gpu_writethrough,
)

__all__ = [
    "field_matvec",
    "field_addmatvec",
    "field_addmatvec_",
    "field_submatvec",
    "field_submatvec_",
    "field_diag",
    "field_adddiag",
    "field_adddiag_",
    "field_subdiag",
    "field_subdiag_",
    "field_kernel",
    "field_addkernel",
    "field_addkernel_",
    "field_subkernel",
    "field_subkernel_",
    "field_relax",
    "field_matvec_rls",
    "field_diag_rls",
    "field_relax_rls",
    "field_precond",
    "field_precond_",
    "field_forward",
    "flow_matvec",
    "flow_addmatvec",
    "flow_addmatvec_",
    "flow_submatvec",
    "flow_submatvec_",
    "flow_diag",
    "flow_adddiag",
    "flow_adddiag_",
    "flow_subdiag",
    "flow_subdiag_",
    "flow_kernel",
    "flow_addkernel",
    "flow_addkernel_",
    "flow_subkernel",
    "flow_subkernel_",
    "flow_relax",
    "flow_matvec_rls",
    "flow_diag_rls",
    "flow_relax_rls",
    "flow_precond",
    "flow_precond_",
    "flow_forward",
]


def _per_channel(value, channels: int, name: str) -> Optional[list]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)] * channels
    out = [float(v) for v in value]
    if len(out) != channels:
        raise ValueError(
            f"{name} must be a scalar or a length-C={channels} sequence"
        )
    return out


def _voxel(value, ndim: int) -> Optional[list]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)] * ndim
    out = [float(v) for v in value]
    if len(out) != ndim:
        raise ValueError(
            f"voxel_size must be a scalar or a length-ndim={ndim} sequence"
        )
    return out


# ---------------------------------------------------------------------------
# In-place accumulate ops (`out (+/-)= ...`)
# ---------------------------------------------------------------------------
#
# One kernel each: the in-place-only C primitive
# `ff::{field,flow}_{matvec,diag}_{add,sub}_`, restored from jitfields'
# `op='+'` / `op='-'` entry points. The out-of-place spelling copies the
# caller's array first and runs the same primitive on the copy.


def _field_matvec_acc(
    inp,
    field,
    absolute,
    membrane,
    bending,
    voxel_size,
    bound,
    ndim,
    sub,
    inplace,
):
    inp = as_gpu_array(inp, name="inp")
    field = as_gpu_array(field, name="field")
    acc = inp if inplace else inp.copy()
    channels = field.shape[-1]
    fn = _ff.field_submatvec_ if sub else _ff.field_addmatvec_
    fn(
        acc,
        field,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return acc


def _field_diag_acc(
    inp, absolute, membrane, bending, voxel_size, bound, ndim, sub, inplace
):
    inp = as_gpu_array(inp, name="inp")
    acc = inp if inplace else inp.copy()
    channels = acc.shape[-1]
    fn = _ff.field_subdiag_ if sub else _ff.field_adddiag_
    fn(
        acc,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return acc


def _field_kernel_acc(
    inp, absolute, membrane, bending, voxel_size, bound, ndim, sub, inplace
):
    inp = as_gpu_array(inp, name="inp")
    acc = inp if inplace else inp.copy()
    channels = acc.shape[-1]
    fn = _ff.field_subkernel_ if sub else _ff.field_addkernel_
    fn(
        acc,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return acc


def _flow_matvec_acc(
    inp,
    flow,
    absolute,
    membrane,
    bending,
    shears,
    div,
    voxel_size,
    bound,
    ndim,
    sub,
    inplace,
):
    inp = as_gpu_array(inp, name="inp")
    flow = as_gpu_array(flow, name="flow")
    acc = inp if inplace else inp.copy()
    fn = _ff.flow_submatvec_ if sub else _ff.flow_addmatvec_
    fn(
        acc,
        flow,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return acc


def _flow_diag_acc(
    inp,
    absolute,
    membrane,
    bending,
    shears,
    div,
    voxel_size,
    bound,
    ndim,
    sub,
    inplace,
):
    inp = as_gpu_array(inp, name="inp")
    acc = inp if inplace else inp.copy()
    fn = _ff.flow_subdiag_ if sub else _ff.flow_adddiag_
    fn(
        acc,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return acc


def _flow_kernel_acc(
    inp,
    absolute,
    membrane,
    bending,
    shears,
    div,
    voxel_size,
    bound,
    ndim,
    sub,
    inplace,
):
    inp = as_gpu_array(inp, name="inp")
    acc = inp if inplace else inp.copy()
    fn = _ff.flow_subkernel_ if sub else _ff.flow_addkernel_
    fn(
        acc,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return acc


# ---------------------------------------------------------------------------
# The regularised system (H + L)
# ---------------------------------------------------------------------------
#
# `*_forward` applies (H + L), `*_precond[_]` the Jacobi approximation of its
# inverse, `*_relax` solves it -- H being the per-voxel data-term Hessian
# (`mat`, compact-symmetric) and L the regulariser. Each is ONE binding,
# `fastfields.dlpack.{field,flow}_{forward,precond,precond_}`.
#
# They used to be assembled here out of `*_matvec` + `sym_matvec` (forward)
# and `*_diag` + `sym_solve` (preconditioner), because the hub entry points
# existed in C++ but had never been bound. That gives the same values, but
# allocates a full-size temporary for the regulariser result and traverses the
# field twice -- and on CUDA every extra temporary is also another block the
# caller has to keep alive until the stream is synchronised. These now call the
# hub entry points.
#
# Leading batch dims of `mat` and the vector are still broadcast, as they were
# through `sym_solve`/`sym_matvec`. The `ndim` spatial axes and the trailing
# channel axis are not: the regulariser's stencil is spatial, so a mismatch
# there is an error rather than something to broadcast (same rule as
# `field_relax`).


def _reg_broadcast(mat, vec, ndim, mat_name="mat", vec_name="vec"):
    """Broadcast the batch dims of a (Hessian, vector) pair for a reg op."""
    mat = as_gpu_array(mat, name=mat_name)
    vec = as_gpu_array(vec, name=vec_name)
    core = int(ndim) + 1
    if mat.ndim < core or vec.ndim < core:
        raise ValueError(
            f"{mat_name} and {vec_name} must have at least ndim+1={core} "
            f"axes (*spatial, C); got {mat.shape} and {vec.shape}"
        )
    batch, (mat_b, vec_b) = broadcast_batch([(mat, core), (vec, core)])
    return batch, mat_b, vec_b


def field_matvec(
    inp: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Apply the field regulariser to a multi-channel field.

    Computes `L @ inp` for the field regulariser `L`, the sum of the requested
    penalties. Each penalty is given per channel and the regulariser never
    couples channels. The highest-order penalty that is set decides the
    finite-difference stencil.

    With unit voxels and only `absolute`, the operator scales each channel by
    its weight; with only `membrane` it is the discrete negative Laplacian.

    The `*_addmatvec`, `*_submatvec` and `*_rls` families accumulate or
    reweight this operator.

    Parameters
    ----------
    inp : cupy.ndarray
        The field to filter, shape `(*batch, *spatial, C)`.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `L @ inp`, a new array the shape of `inp`.

    Raises
    ------
    TypeError
        If `inp` is not a float32 or float64 CuPy array.
    """
    cp = cupy()
    inp = as_gpu_array(inp, name="inp")
    channels = inp.shape[-1]
    out = cp.zeros(inp.shape, dtype=inp.dtype)
    _ff.field_matvec(
        out,
        inp,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return out


def flow_matvec(
    inp: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Apply the flow regulariser to a vector flow field.

    Computes `L @ inp` for the flow regulariser `L`, the sum of the requested
    penalties. The penalties are scalars. A non-zero `shears` or `div` adds the
    linear-elastic terms, which couple the flow components. The highest-order
    penalty that is set decides the finite-difference stencil.

    With unit voxels and only `absolute`, the operator scales each channel by
    its weight; with only `membrane` it is the discrete negative Laplacian.

    The `*_addmatvec`, `*_submatvec` and `*_rls` families accumulate or
    reweight this operator.

    Parameters
    ----------
    inp : cupy.ndarray
        The flow to filter, shape `(*batch, *spatial, ndim)`.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `L @ inp`, a new array the shape of `inp`.

    Raises
    ------
    TypeError
        If `inp` is not a float32 or float64 CuPy array.
    """
    cp = cupy()
    inp = as_gpu_array(inp, name="inp")
    out = cp.zeros(inp.shape, dtype=inp.dtype)
    _ff.flow_matvec(
        out,
        inp,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return out


def _field_channels(channels, *penalties) -> int:
    """Infer C from an explicit value or the per-channel penalty lengths."""
    if channels is not None:
        return int(channels)
    for p in penalties:
        if p is not None and not isinstance(p, (int, float)):
            return len(p)
    return 1


def field_kernel(
    ndim: int,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    channels: int | None = None,
    voxel_size=None,
    bound: int | str = "dct2",
    dtype: Any = None,
) -> Any:
    """Convolution stencil of the field regulariser.

    Returns the small centred kernel that, convolved with a field, reproduces
    [`field_matvec`][fastfields.cupy.field_matvec] with the same penalties. Its
    shape is `(*k, C)`: one independent stencil per channel. `k` is the stencil
    width along each spatial axis: 1 if only `absolute` is set, 3 with
    `membrane`, 5 with `bending`.

    Parameters
    ----------
    ndim : int
        Number of spatial axes of the stencil.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    channels : int, optional
        Number of channels `C`. When omitted it is the length of the first
        penalty given as a sequence, and 1 if there is none.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    dtype : dtype, optional
        Data type of the result.

    Returns
    -------
    cupy.ndarray
        The stencil, with the shape described above.
    """
    cp = cupy()
    ndim = int(ndim)
    channels = _field_channels(channels, absolute, membrane, bending)
    if bending is not None:
        width = 5
    elif membrane is not None:
        width = 3
    else:
        width = 1
    out = cp.zeros(
        tuple([width] * ndim + [channels]), dtype=dtype or cp.float64
    )
    _ff.field_kernel(
        out,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return out


def field_diag(
    shape: Sequence[int],
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    dtype: Any = None,
) -> Any:
    """Diagonal of the field regulariser.

    Builds the diagonal of the field regulariser `L`, which is a cheap Jacobi
    preconditioner. Each penalty is given per channel and the regulariser never
    couples channels.

    The result is allocated from `shape`; for a version that accumulates onto
    an existing array see [`field_adddiag`][fastfields.cupy.field_adddiag].

    Parameters
    ----------
    shape : sequence of int
        Shape of the result, `(*batch, *spatial, C)`.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.
    dtype : dtype, optional
        Data type of the result.

    Returns
    -------
    cupy.ndarray
        The diagonal of `L`, shape `shape`.
    """
    cp = cupy()
    out = cp.zeros(tuple(int(s) for s in shape), dtype=dtype or cp.float64)
    channels = out.shape[-1]
    _ff.field_diag(
        out,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return out


def field_relax(
    field: Any,
    hes: Any,
    grd: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    nb_iter: int = 1,
) -> Any:
    """Solve the regularised system `(H + L) x = g` by relaxation sweeps.

    Runs `nb_iter` Gauss-Seidel sweeps on `(H + L) x = g`, where `H` is the
    per-voxel Hessian `hes` and `L` is the field regulariser `L`. Each penalty
    is given per channel and the regulariser never couples channels.

    `field` is the starting point and receives the result: it is updated in
    place and returned.

    Parameters
    ----------
    field : cupy.ndarray
        Starting point, shape `(*batch, *spatial, C)`. Updated in place.
    hes : cupy.ndarray
        Per-voxel Hessian `H` in compact-symmetric packing, shape `(*batch,
        *spatial, C*(C+1)//2)`.
    grd : cupy.ndarray
        Gradient `g`, same shape as `field`.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.
    nb_iter : int, default=1
        Number of relaxation sweeps.

    Returns
    -------
    cupy.ndarray
        `field` itself, holding the refined solution.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    field = as_gpu_array(field, name="field")
    hes = as_gpu_array(hes, name="hes")
    grd = as_gpu_array(grd, name="grd")
    channels = field.shape[-1]
    _ff.field_relax(
        field,
        hes,
        grd,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        int(nb_iter),
        current_stream_ptr(),
    )
    return field


def field_matvec_rls(
    inp: Any,
    wgt: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Apply the weighted (RLS/JRLS) field regulariser.

    The reweighted form of [`field_matvec`][fastfields.cupy.field_matvec]. The
    weight map `wgt` modulates the strength of the penalties voxel by voxel,
    for edge-preserving or robust regularisation. Its last axis decides the
    mode: 1 for one weight shared by all channels (JRLS), `C` for one per
    channel (RLS).

    Parameters
    ----------
    inp : cupy.ndarray
        The field to filter, shape `(*batch, *spatial, C)`.
    wgt : cupy.ndarray
        Weight map. Its shape is `(*batch, *spatial, 1)` for one weight
        shared by all channels (JRLS) or `(*batch, *spatial, C)` for one
        weight per channel (RLS); the last axis decides which.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        The filtered result, a new array the shape of `inp`.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    cp = cupy()
    inp = as_gpu_array(inp, name="inp")
    wgt = as_gpu_array(wgt, name="wgt")
    channels = inp.shape[-1]
    out = cp.zeros(inp.shape, dtype=inp.dtype)
    _ff.field_matvec_rls(
        out,
        inp,
        wgt,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return out


def field_diag_rls(
    wgt: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    channels: int | None = None,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    dtype: Any = None,
) -> Any:
    """Diagonal of the weighted (RLS/JRLS) field regulariser.

    The diagonal of the operator of
    [`field_matvec_rls`][fastfields.cupy.field_matvec_rls]. Unlike
    [`field_diag`][fastfields.cupy.field_diag] it is shaped by the weight map
    and not by a `shape` argument. The weight map `wgt` modulates the strength
    of the penalties voxel by voxel, for edge-preserving or robust
    regularisation. Its last axis decides the mode: 1 for one weight shared by
    all channels (JRLS), `C` for one per channel (RLS).

    Parameters
    ----------
    wgt : cupy.ndarray
        Weight map. Its shape is `(*batch, *spatial, 1)` for one weight
        shared by all channels (JRLS) or `(*batch, *spatial, C)` for one
        weight per channel (RLS); the last axis decides which.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    channels : int, optional
        Number of channels `C`. When omitted it is the last axis of `wgt`
        if that is not 1, otherwise the length of the first penalty given
        as a sequence, and 1 if there is none.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.
    dtype : dtype, optional
        Data type of the result.

    Returns
    -------
    cupy.ndarray
        The diagonal, shape `(*batch, *spatial, C)`.

    Raises
    ------
    TypeError
        If `wgt` is not a float32 or float64 CuPy array.
    """
    cp = cupy()
    wgt = as_gpu_array(wgt, name="wgt")
    if channels is None:
        channels = (
            wgt.shape[-1]
            if wgt.shape[-1] != 1
            else _field_channels(None, absolute, membrane, bending)
        )
    out = cp.zeros(
        tuple(wgt.shape[:-1]) + (channels,), dtype=dtype or cp.float64
    )
    _ff.field_diag_rls(
        out,
        wgt,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return out


def field_relax_rls(
    field: Any,
    hes: Any,
    grd: Any,
    wgt: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    nb_iter: int = 1,
) -> Any:
    """Solve the weighted system `(H + L(w)) x = g` by relaxation sweeps.

    The reweighted form of [`field_relax`][fastfields.cupy.field_relax]:
    `nb_iter` Gauss-Seidel sweeps on `(H + L(w)) x = g`, where `L(w)` is the
    weighted regulariser of
    [`field_matvec_rls`][fastfields.cupy.field_matvec_rls]. The weight map
    `wgt` modulates the strength of the penalties voxel by voxel, for
    edge-preserving or robust regularisation. Its last axis decides the mode: 1
    for one weight shared by all channels (JRLS), `C` for one per channel
    (RLS).

    `field` is the starting point and receives the result: it is updated in
    place and returned.

    Parameters
    ----------
    field : cupy.ndarray
        Starting point, shape `(*batch, *spatial, C)`. Updated in place.
    hes : cupy.ndarray
        Per-voxel Hessian `H` in compact-symmetric packing, shape `(*batch,
        *spatial, C*(C+1)//2)`.
    grd : cupy.ndarray
        Gradient `g`, same shape as `field`.
    wgt : cupy.ndarray
        Weight map. Its shape is `(*batch, *spatial, 1)` for one weight
        shared by all channels (JRLS) or `(*batch, *spatial, C)` for one
        weight per channel (RLS); the last axis decides which.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.
    nb_iter : int, default=1
        Number of relaxation sweeps.

    Returns
    -------
    cupy.ndarray
        `field` itself, holding the refined solution.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    field = as_gpu_array(field, name="field")
    hes = as_gpu_array(hes, name="hes")
    grd = as_gpu_array(grd, name="grd")
    wgt = as_gpu_array(wgt, name="wgt")
    channels = field.shape[-1]
    _ff.field_relax_rls(
        field,
        hes,
        grd,
        wgt,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        int(nb_iter),
        current_stream_ptr(),
    )
    return field


def flow_diag(
    shape: Sequence[int],
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    dtype: Any = None,
) -> Any:
    """Diagonal of the flow regulariser.

    Builds the diagonal of the flow regulariser `L`, which is a cheap Jacobi
    preconditioner. The penalties are scalars. A non-zero `shears` or `div`
    adds the linear-elastic terms, which couple the flow components.

    The result is allocated from `shape`; for a version that accumulates onto
    an existing array see [`flow_adddiag`][fastfields.cupy.flow_adddiag].

    Parameters
    ----------
    shape : sequence of int
        Shape of the result, `(*batch, *spatial, ndim)`.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.
    dtype : dtype, optional
        Data type of the result.

    Returns
    -------
    cupy.ndarray
        The diagonal of `L`, shape `shape`.
    """
    cp = cupy()
    out = cp.zeros(tuple(int(s) for s in shape), dtype=dtype or cp.float64)
    _ff.flow_diag(
        out,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return out


def flow_kernel(
    ndim: int,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    dtype: Any = None,
) -> Any:
    """Convolution stencil of the flow regulariser.

    Returns the small centred kernel that, convolved with a flow, reproduces
    [`flow_matvec`][fastfields.cupy.flow_matvec] with the same penalties. Its
    shape is `(*k, ndim)` for the per-component stencil, or `(*k, ndim, ndim)`
    when `shears` or `div` select the matrix stencil that couples components.
    `k` is the stencil width along each spatial axis: 1 if only `absolute` is
    set, 3 with `membrane` or the elastic terms, 5 with `bending`.

    Parameters
    ----------
    ndim : int
        Number of spatial axes, which is also the number of flow
        components.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    dtype : dtype, optional
        Data type of the result.

    Returns
    -------
    cupy.ndarray
        The stencil, with the shape described above.
    """
    cp = cupy()
    ndim = int(ndim)
    is_matrix = shears != 0.0 or div != 0.0
    if shears == div == membrane == bending == 0.0:
        width = 1
    elif bending == 0.0:
        width = 3
    else:
        width = 5
    shape = [width] * ndim + [ndim]
    if is_matrix:
        shape += [ndim]
    out = cp.zeros(tuple(shape), dtype=dtype or cp.float64)
    _ff.flow_kernel(
        out,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return out


def flow_relax(
    flow: Any,
    hes: Any,
    grd: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    nb_iter: int = 1,
) -> Any:
    """Solve the regularised system `(H + L) x = g` by relaxation sweeps.

    Runs `nb_iter` Gauss-Seidel sweeps on `(H + L) x = g`, where `H` is the
    per-voxel Hessian `hes` and `L` is the flow regulariser `L`. The penalties
    are scalars. A non-zero `shears` or `div` adds the linear-elastic terms,
    which couple the flow components.

    `flow` is the starting point and receives the result: it is updated in
    place and returned.

    Parameters
    ----------
    flow : cupy.ndarray
        Starting point, shape `(*batch, *spatial, ndim)`. Updated in place.
    hes : cupy.ndarray
        Per-voxel Hessian `H` in compact-symmetric packing, shape `(*batch,
        *spatial, ndim*(ndim+1)//2)`.
    grd : cupy.ndarray
        Gradient `g`, same shape as `flow`.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.
    nb_iter : int, default=1
        Number of relaxation sweeps.

    Returns
    -------
    cupy.ndarray
        `flow` itself, holding the refined solution.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    flow = as_gpu_array(flow, name="flow")
    hes = as_gpu_array(hes, name="hes")
    grd = as_gpu_array(grd, name="grd")
    _ff.flow_relax(
        flow,
        hes,
        grd,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
        int(nb_iter),
        current_stream_ptr(),
    )
    return flow


def flow_matvec_rls(
    inp: Any,
    wgt: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Apply the weighted (RLS/JRLS) flow regulariser.

    The reweighted form of [`flow_matvec`][fastfields.cupy.flow_matvec]. The
    weight map `wgt` modulates the strength of the penalties voxel by voxel,
    for edge-preserving or robust regularisation. Flow weighting is always
    joint: one weight per voxel, shared by all flow components. `bending` is
    not supported with weighting, and a non-zero value is rejected.

    Parameters
    ----------
    inp : cupy.ndarray
        The flow to filter, shape `(*batch, *spatial, ndim)`.
    wgt : cupy.ndarray
        Weight map, shape `(*batch, *spatial, 1)`: one weight per voxel,
        shared by all flow components.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        The filtered result, a new array the shape of `inp`.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    cp = cupy()
    inp = as_gpu_array(inp, name="inp")
    wgt = as_gpu_array(wgt, name="wgt")
    out = cp.zeros(inp.shape, dtype=inp.dtype)
    _ff.flow_matvec_rls(
        out,
        inp,
        wgt,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return out


def flow_diag_rls(
    wgt: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    dtype: Any = None,
) -> Any:
    """Diagonal of the weighted (JRLS) flow regulariser.

    The diagonal of the operator of
    [`flow_matvec_rls`][fastfields.cupy.flow_matvec_rls]. Unlike
    [`flow_diag`][fastfields.cupy.flow_diag] it is shaped by the weight map and
    not by a `shape` argument. The weight map `wgt` modulates the strength of
    the penalties voxel by voxel, for edge-preserving or robust regularisation.
    Flow weighting is always joint: one weight per voxel, shared by all flow
    components. `bending` is not supported with weighting, and a non-zero value
    is rejected.

    Parameters
    ----------
    wgt : cupy.ndarray
        Weight map, shape `(*batch, *spatial, 1)`: one weight per voxel,
        shared by all flow components.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.
    dtype : dtype, optional
        Data type of the result.

    Returns
    -------
    cupy.ndarray
        The diagonal, shape `(*batch, *spatial, ndim)`: the shape of `wgt`
        with its last axis replaced by the number of flow components.

    Raises
    ------
    TypeError
        If `wgt` is not a float32 or float64 CuPy array.
    """
    cp = cupy()
    wgt = as_gpu_array(wgt, name="wgt")
    out = cp.zeros(
        tuple(wgt.shape[:-1]) + (int(ndim),), dtype=dtype or cp.float64
    )
    _ff.flow_diag_rls(
        out,
        wgt,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return out


def flow_relax_rls(
    flow: Any,
    hes: Any,
    grd: Any,
    wgt: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    nb_iter: int = 1,
) -> Any:
    """Solve the weighted system `(H + L(w)) x = g` by relaxation sweeps.

    The reweighted form of [`flow_relax`][fastfields.cupy.flow_relax]:
    `nb_iter` Gauss-Seidel sweeps on `(H + L(w)) x = g`, where `L(w)` is the
    weighted regulariser of
    [`flow_matvec_rls`][fastfields.cupy.flow_matvec_rls]. The weight map `wgt`
    modulates the strength of the penalties voxel by voxel, for edge-preserving
    or robust regularisation. Flow weighting is always joint: one weight per
    voxel, shared by all flow components. `bending` is not supported with
    weighting, and a non-zero value is rejected.

    `flow` is the starting point and receives the result: it is updated in
    place and returned.

    Parameters
    ----------
    flow : cupy.ndarray
        Starting point, shape `(*batch, *spatial, ndim)`. Updated in place.
    hes : cupy.ndarray
        Per-voxel Hessian `H` in compact-symmetric packing, shape `(*batch,
        *spatial, ndim*(ndim+1)//2)`.
    grd : cupy.ndarray
        Gradient `g`, same shape as `flow`.
    wgt : cupy.ndarray
        Weight map, shape `(*batch, *spatial, 1)`: one weight per voxel,
        shared by all flow components.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.
    nb_iter : int, default=1
        Number of relaxation sweeps.

    Returns
    -------
    cupy.ndarray
        `flow` itself, holding the refined solution.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    flow = as_gpu_array(flow, name="flow")
    hes = as_gpu_array(hes, name="hes")
    grd = as_gpu_array(grd, name="grd")
    wgt = as_gpu_array(wgt, name="wgt")
    _ff.flow_relax_rls(
        flow,
        hes,
        grd,
        wgt,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
        int(nb_iter),
        current_stream_ptr(),
    )
    return flow


def flow_precond(
    mat: Any,
    vec: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Jacobi preconditioner: solve `(M + diag(L)) x = vec` voxel by voxel.

    Solves `(M + diag(L)) x = vec` voxel by voxel, where `M` is the per-voxel
    matrix `mat` and `diag(L)` is the diagonal of the flow regulariser `L`. It
    approximates the inverse of the operator of
    [`flow_forward`][fastfields.cupy.flow_forward]. The penalties are scalars.
    A non-zero `shears` or `div` adds the linear-elastic terms, which couple
    the flow components.

    Batch axes of `mat` and `vec` are broadcast against each other; the spatial
    axes and the last axis must match.

    [`flow_precond_`][fastfields.cupy.flow_precond_] is the in-place version.

    Parameters
    ----------
    mat : cupy.ndarray
        Per-voxel matrix `M` in compact-symmetric packing, shape `(*batch,
        *spatial, ndim*(ndim+1)//2)`. Batch axes are broadcast against
        those of `vec`.
    vec : cupy.ndarray
        Right-hand side, shape `(*batch, *spatial, ndim)`. Batch axes are
        broadcast against those of `mat`.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        The solution `x`, shape `(*batch, *spatial, ndim)` with the
        broadcast batch shape.
    """
    cp = cupy()
    batch, mat, vec = _reg_broadcast(mat, vec, ndim)
    core = vec.shape[vec.ndim - int(ndim) - 1 :]
    out = cp.empty(batch + core, dtype=vec.dtype)
    _ff.flow_precond(
        out,
        mat,
        vec,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return out


def flow_forward(
    mat: Any,
    vec: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Apply the regularised system: `(M + L) @ vec`.

    Computes `(M + L) @ vec`, where `M` is the per-voxel matrix `mat` and `L`
    is the flow regulariser `L`. This is the operator that
    [`flow_relax`][fastfields.cupy.flow_relax] solves. The penalties are
    scalars. A non-zero `shears` or `div` adds the linear-elastic terms, which
    couple the flow components.

    Batch axes of `mat` and `vec` are broadcast against each other; the spatial
    axes and the last axis must match.

    Parameters
    ----------
    mat : cupy.ndarray
        Per-voxel matrix `M` in compact-symmetric packing, shape `(*batch,
        *spatial, ndim*(ndim+1)//2)`. Batch axes are broadcast against
        those of `vec`.
    vec : cupy.ndarray
        Vector field, shape `(*batch, *spatial, ndim)`. Batch axes are
        broadcast against those of `mat`.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `(M + L) @ vec`, shape `(*batch, *spatial, ndim)` with the
        broadcast batch shape.
    """
    cp = cupy()
    batch, mat, vec = _reg_broadcast(mat, vec, ndim)
    core = vec.shape[vec.ndim - int(ndim) - 1 :]
    out = cp.empty(batch + core, dtype=vec.dtype)
    _ff.flow_forward(
        out,
        mat,
        vec,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return out


# --- accumulate variants -------------------------------------------------
#
# jitfields' ``_add`` / ``_sub`` (fresh array) and trailing-underscore in-place
# forms, as thin compositions ``inp ± op(...)`` over flow_matvec / flow_diag.


def flow_addmatvec(
    inp: Any,
    flow: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Add the flow regulariser applied to a flow: `inp + L @ flow`.

    Computes `inp + L @ flow` for the flow regulariser `L`. The penalties are
    scalars. A non-zero `shears` or `div` adds the linear-elastic terms, which
    couple the flow components.

    `inp` is copied, never modified;
    [`flow_addmatvec_`][fastfields.cupy.flow_addmatvec_] is the version that
    updates it in place.

    Parameters
    ----------
    inp : cupy.ndarray
        Array the result is accumulated onto, shape `(*batch, *spatial,
        ndim)`.
    flow : cupy.ndarray
        The flow the operator is applied to, shape `(*batch, *spatial,
        ndim)`.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp + L @ flow`, a new array the shape of `inp`.
    """
    return _flow_matvec_acc(
        inp,
        flow,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=False,
    )


def flow_submatvec(
    inp: Any,
    flow: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Subtract the flow regulariser applied to a flow: `inp - L @ flow`.

    Computes `inp - L @ flow` for the flow regulariser `L`. The penalties are
    scalars. A non-zero `shears` or `div` adds the linear-elastic terms, which
    couple the flow components.

    `inp` is copied, never modified;
    [`flow_submatvec_`][fastfields.cupy.flow_submatvec_] is the version that
    updates it in place.

    Parameters
    ----------
    inp : cupy.ndarray
        Array the result is accumulated onto, shape `(*batch, *spatial,
        ndim)`.
    flow : cupy.ndarray
        The flow the operator is applied to, shape `(*batch, *spatial,
        ndim)`.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp - L @ flow`, a new array the shape of `inp`.
    """
    return _flow_matvec_acc(
        inp,
        flow,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=False,
    )


def flow_addmatvec_(
    inp: Any,
    flow: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Accumulate the flow regulariser in place: `inp += L @ flow`.

    Updates `inp` with `inp += L @ flow` for the flow regulariser `L`, in one
    kernel and without a temporary for the product. The penalties are scalars.
    A non-zero `shears` or `div` adds the linear-elastic terms, which couple
    the flow components.

    [`flow_addmatvec`][fastfields.cupy.flow_addmatvec] is the version that
    returns a new array.

    Parameters
    ----------
    inp : cupy.ndarray
        Accumulator, shape `(*batch, *spatial, ndim)`. Updated in place.
    flow : cupy.ndarray
        The flow the operator is applied to, shape `(*batch, *spatial,
        ndim)`.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp` itself, after the update.
    """
    return _flow_matvec_acc(
        inp,
        flow,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=True,
    )


def flow_submatvec_(
    inp: Any,
    flow: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Accumulate the flow regulariser in place: `inp -= L @ flow`.

    Updates `inp` with `inp -= L @ flow` for the flow regulariser `L`, in one
    kernel and without a temporary for the product. The penalties are scalars.
    A non-zero `shears` or `div` adds the linear-elastic terms, which couple
    the flow components.

    [`flow_submatvec`][fastfields.cupy.flow_submatvec] is the version that
    returns a new array.

    Parameters
    ----------
    inp : cupy.ndarray
        Accumulator, shape `(*batch, *spatial, ndim)`. Updated in place.
    flow : cupy.ndarray
        The flow the operator is applied to, shape `(*batch, *spatial,
        ndim)`.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp` itself, after the update.
    """
    return _flow_matvec_acc(
        inp,
        flow,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=True,
    )


def flow_adddiag(
    inp: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Add the diagonal of the flow regulariser: `inp + diag(L)`.

    Computes `inp + diag(L)` for the flow regulariser `L`. The penalties are
    scalars. A non-zero `shears` or `div` adds the linear-elastic terms, which
    couple the flow components.

    `inp` is copied, never modified;
    [`flow_adddiag_`][fastfields.cupy.flow_adddiag_] updates it in place.

    Parameters
    ----------
    inp : cupy.ndarray
        Array the diagonal is accumulated onto, shape `(*batch, *spatial,
        ndim)`.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp + diag(L)`, a new array the shape of `inp`.
    """
    return _flow_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=False,
    )


def flow_subdiag(
    inp: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Subtract the diagonal of the flow regulariser: `inp - diag(L)`.

    Computes `inp - diag(L)` for the flow regulariser `L`. The penalties are
    scalars. A non-zero `shears` or `div` adds the linear-elastic terms, which
    couple the flow components.

    `inp` is copied, never modified;
    [`flow_subdiag_`][fastfields.cupy.flow_subdiag_] updates it in place.

    Parameters
    ----------
    inp : cupy.ndarray
        Array the diagonal is accumulated onto, shape `(*batch, *spatial,
        ndim)`.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp - diag(L)`, a new array the shape of `inp`.
    """
    return _flow_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=False,
    )


def flow_adddiag_(
    inp: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Accumulate the flow regulariser's diagonal in place: `inp += diag(L)`.

    Updates `inp` with `inp += diag(L)` for the flow regulariser `L`, in one
    kernel. The penalties are scalars. A non-zero `shears` or `div` adds the
    linear-elastic terms, which couple the flow components.

    [`flow_adddiag`][fastfields.cupy.flow_adddiag] returns a new array instead.

    Parameters
    ----------
    inp : cupy.ndarray
        Accumulator, shape `(*batch, *spatial, ndim)`. Updated in place.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp` itself, after the update.
    """
    return _flow_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=True,
    )


def flow_subdiag_(
    inp: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Accumulate the flow regulariser's diagonal in place: `inp -= diag(L)`.

    Updates `inp` with `inp -= diag(L)` for the flow regulariser `L`, in one
    kernel. The penalties are scalars. A non-zero `shears` or `div` adds the
    linear-elastic terms, which couple the flow components.

    [`flow_subdiag`][fastfields.cupy.flow_subdiag] returns a new array instead.

    Parameters
    ----------
    inp : cupy.ndarray
        Accumulator, shape `(*batch, *spatial, ndim)`. Updated in place.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp` itself, after the update.
    """
    return _flow_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=True,
    )


def field_precond(
    mat: Any,
    vec: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Jacobi preconditioner: solve `(M + diag(L)) x = vec` voxel by voxel.

    Solves `(M + diag(L)) x = vec` voxel by voxel, where `M` is the per-voxel
    matrix `mat` and `diag(L)` is the diagonal of the field regulariser `L`. It
    approximates the inverse of the operator of
    [`field_forward`][fastfields.cupy.field_forward]. Each penalty is given per
    channel and the regulariser never couples channels.

    Batch axes of `mat` and `vec` are broadcast against each other; the spatial
    axes and the last axis must match.

    [`field_precond_`][fastfields.cupy.field_precond_] is the in-place version.

    Parameters
    ----------
    mat : cupy.ndarray
        Per-voxel matrix `M` in compact-symmetric packing, shape `(*batch,
        *spatial, C*(C+1)//2)`. Batch axes are broadcast against those of
        `vec`.
    vec : cupy.ndarray
        Right-hand side, shape `(*batch, *spatial, C)`. Batch axes are
        broadcast against those of `mat`.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        The solution `x`, shape `(*batch, *spatial, C)` with the broadcast
        batch shape.
    """
    cp = cupy()
    batch, mat, vec = _reg_broadcast(mat, vec, ndim)
    channels = vec.shape[-1]
    core = vec.shape[vec.ndim - int(ndim) - 1 :]
    out = cp.empty(batch + core, dtype=vec.dtype)
    _ff.field_precond(
        out,
        mat,
        vec,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return out


def field_forward(
    mat: Any,
    vec: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Apply the regularised system: `(M + L) @ vec`.

    Computes `(M + L) @ vec`, where `M` is the per-voxel matrix `mat` and `L`
    is the field regulariser `L`. This is the operator that
    [`field_relax`][fastfields.cupy.field_relax] solves. Each penalty is given
    per channel and the regulariser never couples channels.

    Batch axes of `mat` and `vec` are broadcast against each other; the spatial
    axes and the last axis must match.

    Parameters
    ----------
    mat : cupy.ndarray
        Per-voxel matrix `M` in compact-symmetric packing, shape `(*batch,
        *spatial, C*(C+1)//2)`. Batch axes are broadcast against those of
        `vec`.
    vec : cupy.ndarray
        Vector field, shape `(*batch, *spatial, C)`. Batch axes are
        broadcast against those of `mat`.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `(M + L) @ vec`, shape `(*batch, *spatial, C)` with the broadcast
        batch shape.
    """
    cp = cupy()
    batch, mat, vec = _reg_broadcast(mat, vec, ndim)
    channels = vec.shape[-1]
    core = vec.shape[vec.ndim - int(ndim) - 1 :]
    out = cp.empty(batch + core, dtype=vec.dtype)
    _ff.field_forward(
        out,
        mat,
        vec,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return out


def field_addmatvec(
    inp: Any,
    field: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Add the field regulariser applied to a field: `inp + L @ field`.

    Computes `inp + L @ field` for the field regulariser `L`. Each penalty is
    given per channel and the regulariser never couples channels.

    `inp` is copied, never modified;
    [`field_addmatvec_`][fastfields.cupy.field_addmatvec_] is the version that
    updates it in place.

    Parameters
    ----------
    inp : cupy.ndarray
        Array the result is accumulated onto, shape `(*batch, *spatial,
        C)`.
    field : cupy.ndarray
        The field the operator is applied to, shape `(*batch, *spatial,
        C)`.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp + L @ field`, a new array the shape of `inp`.
    """
    return _field_matvec_acc(
        inp,
        field,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=False,
    )


def field_submatvec(
    inp: Any,
    field: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Subtract the field regulariser applied to a field: `inp - L @ field`.

    Computes `inp - L @ field` for the field regulariser `L`. Each penalty is
    given per channel and the regulariser never couples channels.

    `inp` is copied, never modified;
    [`field_submatvec_`][fastfields.cupy.field_submatvec_] is the version that
    updates it in place.

    Parameters
    ----------
    inp : cupy.ndarray
        Array the result is accumulated onto, shape `(*batch, *spatial,
        C)`.
    field : cupy.ndarray
        The field the operator is applied to, shape `(*batch, *spatial,
        C)`.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp - L @ field`, a new array the shape of `inp`.
    """
    return _field_matvec_acc(
        inp,
        field,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=False,
    )


def field_addmatvec_(
    inp: Any,
    field: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Accumulate the field regulariser in place: `inp += L @ field`.

    Updates `inp` with `inp += L @ field` for the field regulariser `L`, in one
    kernel and without a temporary for the product. Each penalty is given per
    channel and the regulariser never couples channels.

    [`field_addmatvec`][fastfields.cupy.field_addmatvec] is the version that
    returns a new array.

    Parameters
    ----------
    inp : cupy.ndarray
        Accumulator, shape `(*batch, *spatial, C)`. Updated in place.
    field : cupy.ndarray
        The field the operator is applied to, shape `(*batch, *spatial,
        C)`.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp` itself, after the update.
    """
    return _field_matvec_acc(
        inp,
        field,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=True,
    )


def field_submatvec_(
    inp: Any,
    field: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Accumulate the field regulariser in place: `inp -= L @ field`.

    Updates `inp` with `inp -= L @ field` for the field regulariser `L`, in one
    kernel and without a temporary for the product. Each penalty is given per
    channel and the regulariser never couples channels.

    [`field_submatvec`][fastfields.cupy.field_submatvec] is the version that
    returns a new array.

    Parameters
    ----------
    inp : cupy.ndarray
        Accumulator, shape `(*batch, *spatial, C)`. Updated in place.
    field : cupy.ndarray
        The field the operator is applied to, shape `(*batch, *spatial,
        C)`.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp` itself, after the update.
    """
    return _field_matvec_acc(
        inp,
        field,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=True,
    )


def field_adddiag(
    inp: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Add the diagonal of the field regulariser: `inp + diag(L)`.

    Computes `inp + diag(L)` for the field regulariser `L`. Each penalty is
    given per channel and the regulariser never couples channels.

    `inp` is copied, never modified;
    [`field_adddiag_`][fastfields.cupy.field_adddiag_] updates it in place.

    Parameters
    ----------
    inp : cupy.ndarray
        Array the diagonal is accumulated onto, shape `(*batch, *spatial,
        C)`.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp + diag(L)`, a new array the shape of `inp`.
    """
    return _field_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=False,
    )


def field_subdiag(
    inp: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Subtract the diagonal of the field regulariser: `inp - diag(L)`.

    Computes `inp - diag(L)` for the field regulariser `L`. Each penalty is
    given per channel and the regulariser never couples channels.

    `inp` is copied, never modified;
    [`field_subdiag_`][fastfields.cupy.field_subdiag_] updates it in place.

    Parameters
    ----------
    inp : cupy.ndarray
        Array the diagonal is accumulated onto, shape `(*batch, *spatial,
        C)`.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp - diag(L)`, a new array the shape of `inp`.
    """
    return _field_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=False,
    )


def field_adddiag_(
    inp: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Accumulate the field regulariser's diagonal in place: `inp += diag(L)`.

    Updates `inp` with `inp += diag(L)` for the field regulariser `L`, in one
    kernel. Each penalty is given per channel and the regulariser never couples
    channels.

    [`field_adddiag`][fastfields.cupy.field_adddiag] returns a new array
    instead.

    Parameters
    ----------
    inp : cupy.ndarray
        Accumulator, shape `(*batch, *spatial, C)`. Updated in place.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp` itself, after the update.
    """
    return _field_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=True,
    )


def field_subdiag_(
    inp: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Accumulate the field regulariser's diagonal in place: `inp -= diag(L)`.

    Updates `inp` with `inp -= diag(L)` for the field regulariser `L`, in one
    kernel. Each penalty is given per channel and the regulariser never couples
    channels.

    [`field_subdiag`][fastfields.cupy.field_subdiag] returns a new array
    instead.

    Parameters
    ----------
    inp : cupy.ndarray
        Accumulator, shape `(*batch, *spatial, C)`. Updated in place.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp` itself, after the update.
    """
    return _field_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=True,
    )


# ---------------------------------------------------------------------------
# In-place preconditioners
# ---------------------------------------------------------------------------


def field_precond_(
    sol: Any,
    mat: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Jacobi preconditioner of the regularised system, in place.

    Solves `(M + diag(L)) x = sol` voxel by voxel and stores `x` in `sol`,
    where `M` is the per-voxel matrix `mat` and `diag(L)` is the diagonal of
    the field regulariser `L`. Each penalty is given per channel and the
    regulariser never couples channels.

    Nothing is broadcast: `sol` is written through, so its shape is the shape
    of the answer. [`field_precond`][fastfields.cupy.field_precond] is the
    version that returns a new array.

    Parameters
    ----------
    sol : cupy.ndarray
        The gradient on entry, the preconditioned solution on exit, shape
        `(*batch, *spatial, C)`. Updated in place.
    mat : cupy.ndarray
        Per-voxel matrix `M` in compact-symmetric packing, shape `(*batch,
        *spatial, C*(C+1)//2)`.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `sol` itself, now holding the solution.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    sol = require_gpu_writethrough(sol, name="sol")
    mat = as_gpu_array(mat, name="mat")
    channels = sol.shape[-1]
    _ff.field_precond_(
        sol,
        mat,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return sol


def flow_precond_(
    sol: Any,
    mat: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Jacobi preconditioner of the regularised system, in place.

    Solves `(M + diag(L)) x = sol` voxel by voxel and stores `x` in `sol`,
    where `M` is the per-voxel matrix `mat` and `diag(L)` is the diagonal of
    the flow regulariser `L`. The penalties are scalars. A non-zero `shears` or
    `div` adds the linear-elastic terms, which couple the flow components.

    Nothing is broadcast: `sol` is written through, so its shape is the shape
    of the answer. [`flow_precond`][fastfields.cupy.flow_precond] is the
    version that returns a new array.

    Parameters
    ----------
    sol : cupy.ndarray
        The gradient on entry, the preconditioned solution on exit, shape
        `(*batch, *spatial, ndim)`. Updated in place.
    mat : cupy.ndarray
        Per-voxel matrix `M` in compact-symmetric packing, shape `(*batch,
        *spatial, ndim*(ndim+1)//2)`.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `sol` itself, now holding the solution.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    sol = require_gpu_writethrough(sol, name="sol")
    mat = as_gpu_array(mat, name="mat")
    _ff.flow_precond_(
        sol,
        mat,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return sol


# ---------------------------------------------------------------------------
# Stencil (kernel) accumulate variants
# ---------------------------------------------------------------------------
#
# The same `inp ± op(...)` pattern as the *_adddiag / *_subdiag family, for
# the materialised Toeplitz stencil rather than the diagonal. `inp` is shaped
# like the output of `field_kernel` / `flow_kernel`.


def field_addkernel(
    inp: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Add the stencil of the field regulariser: `inp + K`.

    Computes `inp + K`, where `K` is the stencil that
    [`field_kernel`][fastfields.cupy.field_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil and is copied, never
    modified; [`field_addkernel_`][fastfields.cupy.field_addkernel_] updates it
    in place.

    Parameters
    ----------
    inp : cupy.ndarray
        Array the stencil is accumulated onto, shaped like the output of
        [`field_kernel`][fastfields.cupy.field_kernel].
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp + K`, a new array the shape of `inp`.
    """
    return _field_kernel_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=False,
    )


def field_subkernel(
    inp: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Subtract the stencil of the field regulariser: `inp - K`.

    Computes `inp - K`, where `K` is the stencil that
    [`field_kernel`][fastfields.cupy.field_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil and is copied, never
    modified; [`field_subkernel_`][fastfields.cupy.field_subkernel_] updates it
    in place.

    Parameters
    ----------
    inp : cupy.ndarray
        Array the stencil is accumulated onto, shaped like the output of
        [`field_kernel`][fastfields.cupy.field_kernel].
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp - K`, a new array the shape of `inp`.
    """
    return _field_kernel_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=False,
    )


def field_addkernel_(
    inp: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Accumulate the field regulariser's stencil in place: `inp += K`.

    Updates `inp` with `inp += K`, where `K` is the stencil that
    [`field_kernel`][fastfields.cupy.field_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil.
    [`field_addkernel`][fastfields.cupy.field_addkernel] returns a new array
    instead.

    Parameters
    ----------
    inp : cupy.ndarray
        Accumulator, shaped like the output of
        [`field_kernel`][fastfields.cupy.field_kernel]. Updated in place.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp` itself, after the update.
    """
    return _field_kernel_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=True,
    )


def field_subkernel_(
    inp: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Accumulate the field regulariser's stencil in place: `inp -= K`.

    Updates `inp` with `inp -= K`, where `K` is the stencil that
    [`field_kernel`][fastfields.cupy.field_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil.
    [`field_subkernel`][fastfields.cupy.field_subkernel] returns a new array
    instead.

    Parameters
    ----------
    inp : cupy.ndarray
        Accumulator, shaped like the output of
        [`field_kernel`][fastfields.cupy.field_kernel]. Updated in place.
    absolute : float or sequence of float, optional
        Weight of the absolute (L2) penalty. A scalar applies to every
        channel; a sequence gives one weight per channel. `None` disables
        the penalty.
    membrane : float or sequence of float, optional
        Weight of the membrane (first-order) penalty, in the same format as
        `absolute`.
    bending : float or sequence of float, optional
        Weight of the bending (second-order) penalty, in the same format as
        `absolute`.
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp` itself, after the update.
    """
    return _field_kernel_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=True,
    )


def flow_addkernel(
    inp: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Add the stencil of the flow regulariser: `inp + K`.

    Computes `inp + K`, where `K` is the stencil that
    [`flow_kernel`][fastfields.cupy.flow_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil and is copied, never
    modified; [`flow_addkernel_`][fastfields.cupy.flow_addkernel_] updates it
    in place.

    Parameters
    ----------
    inp : cupy.ndarray
        Array the stencil is accumulated onto, shaped like the output of
        [`flow_kernel`][fastfields.cupy.flow_kernel].
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp + K`, a new array the shape of `inp`.
    """
    return _flow_kernel_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=False,
    )


def flow_subkernel(
    inp: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Subtract the stencil of the flow regulariser: `inp - K`.

    Computes `inp - K`, where `K` is the stencil that
    [`flow_kernel`][fastfields.cupy.flow_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil and is copied, never
    modified; [`flow_subkernel_`][fastfields.cupy.flow_subkernel_] updates it
    in place.

    Parameters
    ----------
    inp : cupy.ndarray
        Array the stencil is accumulated onto, shaped like the output of
        [`flow_kernel`][fastfields.cupy.flow_kernel].
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp - K`, a new array the shape of `inp`.
    """
    return _flow_kernel_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=False,
    )


def flow_addkernel_(
    inp: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Accumulate the flow regulariser's stencil in place: `inp += K`.

    Updates `inp` with `inp += K`, where `K` is the stencil that
    [`flow_kernel`][fastfields.cupy.flow_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil.
    [`flow_addkernel`][fastfields.cupy.flow_addkernel] returns a new array
    instead.

    Parameters
    ----------
    inp : cupy.ndarray
        Accumulator, shaped like the output of
        [`flow_kernel`][fastfields.cupy.flow_kernel]. Updated in place.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp` itself, after the update.
    """
    return _flow_kernel_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=True,
    )


def flow_subkernel_(
    inp: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Accumulate the flow regulariser's stencil in place: `inp -= K`.

    Updates `inp` with `inp -= K`, where `K` is the stencil that
    [`flow_kernel`][fastfields.cupy.flow_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil.
    [`flow_subkernel`][fastfields.cupy.flow_subkernel] returns a new array
    instead.

    Parameters
    ----------
    inp : cupy.ndarray
        Accumulator, shaped like the output of
        [`flow_kernel`][fastfields.cupy.flow_kernel]. Updated in place.
    absolute : float, default=0.0
        Weight of the absolute (L2) penalty.
    membrane : float, default=0.0
        Weight of the membrane (first-order) penalty.
    bending : float, default=0.0
        Weight of the bending (second-order) penalty.
    shears : float, default=0.0
        Weight of the linear-elastic shear penalty (Lamé `mu`). Together
        with `div` it couples the flow components.
    div : float, default=0.0
        Weight of the linear-elastic divergence penalty (Lamé `lambda`).
    voxel_size : float or sequence of float, optional
        Spatial voxel size: one value per spatial axis, or a single value
        for all. `None` means unit voxels.
    bound : int or str, default="dct2"
        Boundary condition: an integer from 0 to 7, a
        [`Bound`][fastfields.helpers.Bound] member, or a name such as
        `"dct2"` or `"wrap"`.
    ndim : int, default=1
        Number of spatial axes. They are the `ndim` axes just before the
        channel axis; this is not inferred, so pass it explicitly for
        anything but 1D data.

    Returns
    -------
    cupy.ndarray
        `inp` itself, after the update.
    """
    return _flow_kernel_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=True,
    )
