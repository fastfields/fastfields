"""Spatial regularisers — numpy.

Two operator families over the last ``ndim`` spatial axes:

* **field** — a multi-channel field ``(*batch, *spatial, C)``. The
  ``absolute`` / ``membrane`` / ``bending`` penalties are **per-channel**
  (a scalar broadcasts to all ``C`` channels, or pass a length-``C`` sequence).
* **flow** — a vector flow field. The penalties are **scalars**.

Each family offers the operator (``*_matvec``, apply the regulariser) and its
diagonal (``*_diag``, a preconditioner). With ``voxel_size=None`` all voxels
are unit size; with only ``absolute`` the operator is a per-channel scaling.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import fastfields.dlpack as _ff

import numpy as np

from ._util import _as_bound, _as_float_array, _broadcast_batch

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
    "field_cg",
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
    "CGInfo",
]


def _per_channel(
    value: float | Sequence[float] | None, channels: int, name: str
) -> Optional[list]:
    """Normalise a per-channel penalty to a length-``channels`` list/None."""
    if value is None:
        return None
    if np.isscalar(value):
        return [float(value)] * channels
    out = [float(v) for v in value]
    if len(out) != channels:
        raise ValueError(
            f"{name} must be a scalar or a length-C={channels} sequence, "
            f"got {value!r}"
        )
    return out


def _voxel_size(
    value: float | Sequence[float] | None, ndim: int
) -> Optional[list]:
    if value is None:
        return None
    if np.isscalar(value):
        return [float(value)] * ndim
    out = [float(v) for v in value]
    if len(out) != ndim:
        raise ValueError(
            f"voxel_size must be a scalar or a length-ndim={ndim} sequence, "
            f"got {value!r}"
        )
    return out


# ---------------------------------------------------------------------------
# In-place accumulate ops (`out (+/-)= ...`)
# ---------------------------------------------------------------------------
#
# There is exactly ONE kernel behind each of these: the in-place-only C
# primitive `ff::{field,flow}_{matvec,diag}_{add,sub}_`, restored from
# jitfields' `op='+'` / `op='-'` entry points. The out-of-place spelling simply
# copies the caller's array first and then runs the same primitive on the copy
# -- it is not a second code path, and it never materialises a separate
# temporary for the regulariser result.


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
    inp = _as_float_array(inp, "inp")
    field = _as_float_array(field, "field")
    acc = inp if inplace else inp.copy()
    channels = field.shape[-1]
    fn = _ff.field_submatvec_ if sub else _ff.field_addmatvec_
    fn(
        acc,
        field,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return acc


def _field_diag_acc(
    inp, absolute, membrane, bending, voxel_size, bound, ndim, sub, inplace
):
    inp = _as_float_array(inp, "inp")
    acc = inp if inplace else inp.copy()
    channels = acc.shape[-1]
    fn = _ff.field_subdiag_ if sub else _ff.field_adddiag_
    fn(
        acc,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return acc


# ---------------------------------------------------------------------------
# The regularised system (H + L)
# ---------------------------------------------------------------------------
#
# `*_forward` applies (H + L), `*_precond[_]` applies the Jacobi approximation
# of its inverse, and `*_relax` / `field_cg` solve it -- where H is the
# per-voxel data-term Hessian (`mat`, compact-symmetric) and L the
# regulariser. Each of these is ONE binding, `fastfields.dlpack.{field,flow}_
# {forward,precond,precond_}`.
#
# They used to be assembled here in Python out of `*_matvec` + `sym_matvec`
# (forward) and `*_diag` + `sym_solve` (preconditioner), because the hub
# entry points existed in C++ but had never been bound. That composition
# computes the same values; it just materialises a full-size temporary for the
# regulariser result and walks the field twice. The hub entry points are now
# bound, so these call them.
#
# Batch (leading) dims of `mat` and the vector are still broadcast, as they
# were through `sym_solve`/`sym_matvec`. The *spatial* axes and the channel
# axis are not: the regulariser's stencil is spatial, so a mismatch there is
# an error rather than something to broadcast (same rule as `field_relax`).
# `ndim + 1` trailing axes are therefore the core shape.


def _reg_broadcast(mat, vec, ndim, mat_name="mat", vec_name="vec"):
    """Broadcast the batch dims of a (Hessian, vector) pair for a reg op."""
    mat = _as_float_array(mat, mat_name)
    vec = _as_float_array(vec, vec_name)
    core = int(ndim) + 1
    if mat.ndim < core or vec.ndim < core:
        raise ValueError(
            f"{mat_name} and {vec_name} must have at least ndim+1={core} "
            f"axes (*spatial, C); got {mat.shape} and {vec.shape}"
        )
    batch, (mat_b, vec_b) = _broadcast_batch([(mat, core), (vec, core)])
    return batch, mat_b, vec_b


def _field_kernel_acc(
    inp, absolute, membrane, bending, voxel_size, bound, ndim, sub, inplace
):
    inp = _as_float_array(inp, "inp")
    acc = inp if inplace else inp.copy()
    channels = acc.shape[-1]
    fn = _ff.field_subkernel_ if sub else _ff.field_addkernel_
    fn(
        acc,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
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
    inp = _as_float_array(inp, "inp")
    flow = _as_float_array(flow, "flow")
    acc = inp if inplace else inp.copy()
    fn = _ff.flow_submatvec_ if sub else _ff.flow_addmatvec_
    fn(
        acc,
        flow,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
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
    inp = _as_float_array(inp, "inp")
    acc = inp if inplace else inp.copy()
    fn = _ff.flow_subdiag_ if sub else _ff.flow_adddiag_
    fn(
        acc,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
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
    inp = _as_float_array(inp, "inp")
    acc = inp if inplace else inp.copy()
    fn = _ff.flow_subkernel_ if sub else _ff.flow_addkernel_
    fn(
        acc,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return acc


def field_matvec(
    inp: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
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
    inp : numpy.ndarray
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
    numpy.ndarray
        `L @ inp`, a new array the shape of `inp`.
    """
    inp = _as_float_array(inp, "inp")
    channels = inp.shape[-1]
    out = np.zeros_like(inp)
    _ff.field_matvec(
        out,
        inp,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def _field_channels(
    channels: int | None,
    *penalties: float | Sequence[float] | None,
) -> int:
    """Infer the channel count ``C`` from an explicit value or penalty lengths.

    A per-channel penalty passed as a sequence fixes ``C`` (its length); scalar
    / None penalties broadcast. Defaults to 1 if nothing pins it down.
    """
    if channels is not None:
        return int(channels)
    for p in penalties:
        if p is not None and not np.isscalar(p):
            return len(p)
    return 1


def field_kernel(
    ndim: int,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    channels: int | None = None,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Convolution stencil of the field regulariser.

    Returns the small centred kernel that, convolved with a field, reproduces
    [`field_matvec`][fastfields.numpy.field_matvec] with the same penalties.
    Its shape is `(*k, C)`: one independent stencil per channel. `k` is the
    stencil width along each spatial axis: 1 if only `absolute` is set, 3 with
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
    dtype : dtype, default=numpy.float64
        Data type of the result.

    Returns
    -------
    numpy.ndarray
        The stencil, with the shape described above.
    """
    ndim = int(ndim)
    channels = _field_channels(channels, absolute, membrane, bending)
    if bending is not None:
        width = 5
    elif membrane is not None:
        width = 3
    else:
        width = 1
    out = np.zeros([width] * ndim + [channels], dtype=dtype)
    _ff.field_kernel(
        out,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def field_diag(
    shape: Sequence[int],
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Diagonal of the field regulariser.

    Builds the diagonal of the field regulariser `L`, which is a cheap Jacobi
    preconditioner. Each penalty is given per channel and the regulariser never
    couples channels.

    The result is allocated from `shape`; for a version that accumulates onto
    an existing array see [`field_adddiag`][fastfields.numpy.field_adddiag].

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
    dtype : dtype, default=numpy.float64
        Data type of the result.

    Returns
    -------
    numpy.ndarray
        The diagonal of `L`, shape `shape`.
    """
    out = np.zeros(tuple(int(s) for s in shape), dtype=dtype)
    channels = out.shape[-1]
    _ff.field_diag(
        out,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def field_relax(
    field: np.ndarray,
    hes: np.ndarray,
    grd: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
    nb_iter: int = 1,
) -> np.ndarray:
    """Solve the regularised system `(H + L) x = g` by relaxation sweeps.

    Runs `nb_iter` Gauss-Seidel sweeps on `(H + L) x = g`, where `H` is the
    per-voxel Hessian `hes` and `L` is the field regulariser `L`. Each penalty
    is given per channel and the regulariser never couples channels.

    `field` is the starting point and receives the result: it is updated in
    place and returned.

    Parameters
    ----------
    field : numpy.ndarray
        Starting point, shape `(*batch, *spatial, C)`. Updated in place.
        Must be a writable float32 or float64 array: anything else is
        converted to a copy, and your array is left unchanged.
    hes : numpy.ndarray
        Per-voxel Hessian `H` in compact-symmetric packing, shape `(*batch,
        *spatial, C*(C+1)//2)`.
    grd : numpy.ndarray
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
    numpy.ndarray
        `field` itself, holding the refined solution.
    """
    field = _as_float_array(field, "field")
    hes = _as_float_array(hes, "hes")
    grd = _as_float_array(grd, "grd")
    channels = field.shape[-1]
    _ff.field_relax(
        field,
        hes,
        grd,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
        nb_iter=int(nb_iter),
    )
    return field


def field_matvec_rls(
    inp: np.ndarray,
    wgt: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Apply the weighted (RLS/JRLS) field regulariser.

    The reweighted form of [`field_matvec`][fastfields.numpy.field_matvec]. The
    weight map `wgt` modulates the strength of the penalties voxel by voxel,
    for edge-preserving or robust regularisation. Its last axis decides the
    mode: 1 for one weight shared by all channels (JRLS), `C` for one per
    channel (RLS).

    Parameters
    ----------
    inp : numpy.ndarray
        The field to filter, shape `(*batch, *spatial, C)`.
    wgt : numpy.ndarray
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
    numpy.ndarray
        The filtered result, a new array the shape of `inp`.
    """
    inp = _as_float_array(inp, "inp")
    wgt = _as_float_array(wgt, "wgt")
    channels = inp.shape[-1]
    out = np.zeros_like(inp)
    _ff.field_matvec_rls(
        out,
        inp,
        wgt,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def field_diag_rls(
    wgt: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    channels: int | None = None,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Diagonal of the weighted (RLS/JRLS) field regulariser.

    The diagonal of the operator of
    [`field_matvec_rls`][fastfields.numpy.field_matvec_rls]. Unlike
    [`field_diag`][fastfields.numpy.field_diag] it is shaped by the weight map
    and not by a `shape` argument. The weight map `wgt` modulates the strength
    of the penalties voxel by voxel, for edge-preserving or robust
    regularisation. Its last axis decides the mode: 1 for one weight shared by
    all channels (JRLS), `C` for one per channel (RLS).

    Parameters
    ----------
    wgt : numpy.ndarray
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
    dtype : dtype, default=numpy.float64
        Data type of the result.

    Returns
    -------
    numpy.ndarray
        The diagonal, shape `(*batch, *spatial, C)`.
    """
    wgt = _as_float_array(wgt, "wgt")
    if channels is None:
        channels = (
            wgt.shape[-1]
            if wgt.shape[-1] != 1
            else _field_channels(None, absolute, membrane, bending)
        )
    out = np.zeros(wgt.shape[:-1] + (channels,), dtype=dtype)
    _ff.field_diag_rls(
        out,
        wgt,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def field_relax_rls(
    field: np.ndarray,
    hes: np.ndarray,
    grd: np.ndarray,
    wgt: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
    nb_iter: int = 1,
) -> np.ndarray:
    """Solve the weighted system `(H + L(w)) x = g` by relaxation sweeps.

    The reweighted form of [`field_relax`][fastfields.numpy.field_relax]:
    `nb_iter` Gauss-Seidel sweeps on `(H + L(w)) x = g`, where `L(w)` is the
    weighted regulariser of
    [`field_matvec_rls`][fastfields.numpy.field_matvec_rls]. The weight map
    `wgt` modulates the strength of the penalties voxel by voxel, for
    edge-preserving or robust regularisation. Its last axis decides the mode: 1
    for one weight shared by all channels (JRLS), `C` for one per channel
    (RLS).

    `field` is the starting point and receives the result: it is updated in
    place and returned.

    Parameters
    ----------
    field : numpy.ndarray
        Starting point, shape `(*batch, *spatial, C)`. Updated in place.
        Must be a writable float32 or float64 array: anything else is
        converted to a copy, and your array is left unchanged.
    hes : numpy.ndarray
        Per-voxel Hessian `H` in compact-symmetric packing, shape `(*batch,
        *spatial, C*(C+1)//2)`.
    grd : numpy.ndarray
        Gradient `g`, same shape as `field`.
    wgt : numpy.ndarray
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
    numpy.ndarray
        `field` itself, holding the refined solution.
    """
    field = _as_float_array(field, "field")
    hes = _as_float_array(hes, "hes")
    grd = _as_float_array(grd, "grd")
    wgt = _as_float_array(wgt, "wgt")
    channels = field.shape[-1]
    _ff.field_relax_rls(
        field,
        hes,
        grd,
        wgt,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
        nb_iter=int(nb_iter),
    )
    return field


def flow_matvec(
    inp: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
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
    inp : numpy.ndarray
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
    numpy.ndarray
        `L @ inp`, a new array the shape of `inp`.
    """
    inp = _as_float_array(inp, "inp")
    out = np.zeros_like(inp)
    _ff.flow_matvec(
        out,
        inp,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def flow_diag(
    shape: Sequence[int],
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Diagonal of the flow regulariser.

    Builds the diagonal of the flow regulariser `L`, which is a cheap Jacobi
    preconditioner. The penalties are scalars. A non-zero `shears` or `div`
    adds the linear-elastic terms, which couple the flow components.

    The result is allocated from `shape`; for a version that accumulates onto
    an existing array see [`flow_adddiag`][fastfields.numpy.flow_adddiag].

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
    dtype : dtype, default=numpy.float64
        Data type of the result.

    Returns
    -------
    numpy.ndarray
        The diagonal of `L`, shape `shape`.
    """
    out = np.zeros(tuple(int(s) for s in shape), dtype=dtype)
    _ff.flow_diag(
        out,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
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
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Convolution stencil of the flow regulariser.

    Returns the small centred kernel that, convolved with a flow, reproduces
    [`flow_matvec`][fastfields.numpy.flow_matvec] with the same penalties. Its
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
    dtype : dtype, default=numpy.float64
        Data type of the result.

    Returns
    -------
    numpy.ndarray
        The stencil, with the shape described above.
    """
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
    out = np.zeros(shape, dtype=dtype)
    _ff.flow_kernel(
        out,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def flow_relax(
    flow: np.ndarray,
    hes: np.ndarray,
    grd: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
    nb_iter: int = 1,
) -> np.ndarray:
    """Solve the regularised system `(H + L) x = g` by relaxation sweeps.

    Runs `nb_iter` Gauss-Seidel sweeps on `(H + L) x = g`, where `H` is the
    per-voxel Hessian `hes` and `L` is the flow regulariser `L`. The penalties
    are scalars. A non-zero `shears` or `div` adds the linear-elastic terms,
    which couple the flow components.

    `flow` is the starting point and receives the result: it is updated in
    place and returned.

    Parameters
    ----------
    flow : numpy.ndarray
        Starting point, shape `(*batch, *spatial, ndim)`. Updated in place.
        Must be a writable float32 or float64 array: anything else is
        converted to a copy, and your array is left unchanged.
    hes : numpy.ndarray
        Per-voxel Hessian `H` in compact-symmetric packing, shape `(*batch,
        *spatial, ndim*(ndim+1)//2)`.
    grd : numpy.ndarray
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
    numpy.ndarray
        `flow` itself, holding the refined solution.
    """
    flow = _as_float_array(flow, "flow")
    hes = _as_float_array(hes, "hes")
    grd = _as_float_array(grd, "grd")
    _ff.flow_relax(
        flow,
        hes,
        grd,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
        nb_iter=int(nb_iter),
    )
    return flow


def flow_matvec_rls(
    inp: np.ndarray,
    wgt: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Apply the weighted (RLS/JRLS) flow regulariser.

    The reweighted form of [`flow_matvec`][fastfields.numpy.flow_matvec]. The
    weight map `wgt` modulates the strength of the penalties voxel by voxel,
    for edge-preserving or robust regularisation. Flow weighting is always
    joint: one weight per voxel, shared by all flow components. `bending` is
    not supported with weighting, and a non-zero value is rejected.

    Parameters
    ----------
    inp : numpy.ndarray
        The flow to filter, shape `(*batch, *spatial, ndim)`.
    wgt : numpy.ndarray
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
    numpy.ndarray
        The filtered result, a new array the shape of `inp`.
    """
    inp = _as_float_array(inp, "inp")
    wgt = _as_float_array(wgt, "wgt")
    out = np.zeros_like(inp)
    _ff.flow_matvec_rls(
        out,
        inp,
        wgt,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def flow_diag_rls(
    wgt: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Diagonal of the weighted (JRLS) flow regulariser.

    The diagonal of the operator of
    [`flow_matvec_rls`][fastfields.numpy.flow_matvec_rls]. Unlike
    [`flow_diag`][fastfields.numpy.flow_diag] it is shaped by the weight map
    and not by a `shape` argument. The weight map `wgt` modulates the strength
    of the penalties voxel by voxel, for edge-preserving or robust
    regularisation. Flow weighting is always joint: one weight per voxel,
    shared by all flow components. `bending` is not supported with weighting,
    and a non-zero value is rejected.

    Parameters
    ----------
    wgt : numpy.ndarray
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
    dtype : dtype, default=numpy.float64
        Data type of the result.

    Returns
    -------
    numpy.ndarray
        The diagonal, shape `(*batch, *spatial, ndim)`: the shape of `wgt`
        with its last axis replaced by the number of flow components.
    """
    wgt = _as_float_array(wgt, "wgt")
    out = np.zeros(wgt.shape[:-1] + (ndim,), dtype=dtype)
    _ff.flow_diag_rls(
        out,
        wgt,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def flow_relax_rls(
    flow: np.ndarray,
    hes: np.ndarray,
    grd: np.ndarray,
    wgt: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
    nb_iter: int = 1,
) -> np.ndarray:
    """Solve the weighted system `(H + L(w)) x = g` by relaxation sweeps.

    The reweighted form of [`flow_relax`][fastfields.numpy.flow_relax]:
    `nb_iter` Gauss-Seidel sweeps on `(H + L(w)) x = g`, where `L(w)` is the
    weighted regulariser of
    [`flow_matvec_rls`][fastfields.numpy.flow_matvec_rls]. The weight map `wgt`
    modulates the strength of the penalties voxel by voxel, for edge-preserving
    or robust regularisation. Flow weighting is always joint: one weight per
    voxel, shared by all flow components. `bending` is not supported with
    weighting, and a non-zero value is rejected.

    `flow` is the starting point and receives the result: it is updated in
    place and returned.

    Parameters
    ----------
    flow : numpy.ndarray
        Starting point, shape `(*batch, *spatial, ndim)`. Updated in place.
        Must be a writable float32 or float64 array: anything else is
        converted to a copy, and your array is left unchanged.
    hes : numpy.ndarray
        Per-voxel Hessian `H` in compact-symmetric packing, shape `(*batch,
        *spatial, ndim*(ndim+1)//2)`.
    grd : numpy.ndarray
        Gradient `g`, same shape as `flow`.
    wgt : numpy.ndarray
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
    numpy.ndarray
        `flow` itself, holding the refined solution.
    """
    flow = _as_float_array(flow, "flow")
    hes = _as_float_array(hes, "hes")
    grd = _as_float_array(grd, "grd")
    wgt = _as_float_array(wgt, "wgt")
    _ff.flow_relax_rls(
        flow,
        hes,
        grd,
        wgt,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
        nb_iter=int(nb_iter),
    )
    return flow


def flow_precond(
    mat: np.ndarray,
    vec: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Jacobi preconditioner: solve `(M + diag(L)) x = vec` voxel by voxel.

    Solves `(M + diag(L)) x = vec` voxel by voxel, where `M` is the per-voxel
    matrix `mat` and `diag(L)` is the diagonal of the flow regulariser `L`. It
    approximates the inverse of the operator of
    [`flow_forward`][fastfields.numpy.flow_forward]. The penalties are scalars.
    A non-zero `shears` or `div` adds the linear-elastic terms, which couple
    the flow components.

    Batch axes of `mat` and `vec` are broadcast against each other; the spatial
    axes and the last axis must match.

    [`flow_precond_`][fastfields.numpy.flow_precond_] is the in-place version.

    Parameters
    ----------
    mat : numpy.ndarray
        Per-voxel matrix `M` in compact-symmetric packing, shape `(*batch,
        *spatial, ndim*(ndim+1)//2)`. Batch axes are broadcast against
        those of `vec`.
    vec : numpy.ndarray
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
    numpy.ndarray
        The solution `x`, shape `(*batch, *spatial, ndim)` with the
        broadcast batch shape.
    """
    batch, mat, vec = _reg_broadcast(mat, vec, ndim)
    out = np.empty(batch + vec.shape[vec.ndim - int(ndim) - 1 :], vec.dtype)
    _ff.flow_precond(
        out,
        mat,
        vec,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def flow_forward(
    mat: np.ndarray,
    vec: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Apply the regularised system: `(M + L) @ vec`.

    Computes `(M + L) @ vec`, where `M` is the per-voxel matrix `mat` and `L`
    is the flow regulariser `L`. This is the operator that
    [`flow_relax`][fastfields.numpy.flow_relax] solves. The penalties are
    scalars. A non-zero `shears` or `div` adds the linear-elastic terms, which
    couple the flow components.

    Batch axes of `mat` and `vec` are broadcast against each other; the spatial
    axes and the last axis must match.

    Parameters
    ----------
    mat : numpy.ndarray
        Per-voxel matrix `M` in compact-symmetric packing, shape `(*batch,
        *spatial, ndim*(ndim+1)//2)`. Batch axes are broadcast against
        those of `vec`.
    vec : numpy.ndarray
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
    numpy.ndarray
        `(M + L) @ vec`, shape `(*batch, *spatial, ndim)` with the
        broadcast batch shape.
    """
    batch, mat, vec = _reg_broadcast(mat, vec, ndim)
    out = np.empty(batch + vec.shape[vec.ndim - int(ndim) - 1 :], vec.dtype)
    _ff.flow_forward(
        out,
        mat,
        vec,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


# --- accumulate variants -------------------------------------------------
#
# jitfields exposes ``_add`` / ``_sub`` (write a fresh array) and trailing-
# underscore in-place (``_add_`` / ``_sub_``) forms of the flow operator and
# its diagonal. They are thin compositions over `flow_matvec` /
# `flow_diag` — ``out = inp ± op(...)`` — so no new kernel is needed.


def flow_addmatvec(
    inp: np.ndarray,
    flow: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Add the flow regulariser applied to a flow: `inp + L @ flow`.

    Computes `inp + L @ flow` for the flow regulariser `L`. The penalties are
    scalars. A non-zero `shears` or `div` adds the linear-elastic terms, which
    couple the flow components.

    `inp` is copied, never modified;
    [`flow_addmatvec_`][fastfields.numpy.flow_addmatvec_] is the version that
    updates it in place.

    Parameters
    ----------
    inp : numpy.ndarray
        Array the result is accumulated onto, shape `(*batch, *spatial,
        ndim)`.
    flow : numpy.ndarray
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
    numpy.ndarray
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
    inp: np.ndarray,
    flow: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Subtract the flow regulariser applied to a flow: `inp - L @ flow`.

    Computes `inp - L @ flow` for the flow regulariser `L`. The penalties are
    scalars. A non-zero `shears` or `div` adds the linear-elastic terms, which
    couple the flow components.

    `inp` is copied, never modified;
    [`flow_submatvec_`][fastfields.numpy.flow_submatvec_] is the version that
    updates it in place.

    Parameters
    ----------
    inp : numpy.ndarray
        Array the result is accumulated onto, shape `(*batch, *spatial,
        ndim)`.
    flow : numpy.ndarray
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
    numpy.ndarray
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
    inp: np.ndarray,
    flow: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Accumulate the flow regulariser in place: `inp += L @ flow`.

    Updates `inp` with `inp += L @ flow` for the flow regulariser `L`, in one
    kernel and without a temporary for the product. The penalties are scalars.
    A non-zero `shears` or `div` adds the linear-elastic terms, which couple
    the flow components.

    [`flow_addmatvec`][fastfields.numpy.flow_addmatvec] is the version that
    returns a new array.

    Parameters
    ----------
    inp : numpy.ndarray
        Accumulator, shape `(*batch, *spatial, ndim)`. Updated in place.
        Must be a writable float32 or float64 array: anything else is
        converted to a copy, and your array is left unchanged.
    flow : numpy.ndarray
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
    numpy.ndarray
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
    inp: np.ndarray,
    flow: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Accumulate the flow regulariser in place: `inp -= L @ flow`.

    Updates `inp` with `inp -= L @ flow` for the flow regulariser `L`, in one
    kernel and without a temporary for the product. The penalties are scalars.
    A non-zero `shears` or `div` adds the linear-elastic terms, which couple
    the flow components.

    [`flow_submatvec`][fastfields.numpy.flow_submatvec] is the version that
    returns a new array.

    Parameters
    ----------
    inp : numpy.ndarray
        Accumulator, shape `(*batch, *spatial, ndim)`. Updated in place.
        Must be a writable float32 or float64 array: anything else is
        converted to a copy, and your array is left unchanged.
    flow : numpy.ndarray
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
    numpy.ndarray
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
    inp: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Add the diagonal of the flow regulariser: `inp + diag(L)`.

    Computes `inp + diag(L)` for the flow regulariser `L`. The penalties are
    scalars. A non-zero `shears` or `div` adds the linear-elastic terms, which
    couple the flow components.

    `inp` is copied, never modified;
    [`flow_adddiag_`][fastfields.numpy.flow_adddiag_] updates it in place.

    Parameters
    ----------
    inp : numpy.ndarray
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
    numpy.ndarray
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
    inp: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Subtract the diagonal of the flow regulariser: `inp - diag(L)`.

    Computes `inp - diag(L)` for the flow regulariser `L`. The penalties are
    scalars. A non-zero `shears` or `div` adds the linear-elastic terms, which
    couple the flow components.

    `inp` is copied, never modified;
    [`flow_subdiag_`][fastfields.numpy.flow_subdiag_] updates it in place.

    Parameters
    ----------
    inp : numpy.ndarray
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
    numpy.ndarray
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
    inp: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Accumulate the flow regulariser's diagonal in place: `inp += diag(L)`.

    Updates `inp` with `inp += diag(L)` for the flow regulariser `L`, in one
    kernel. The penalties are scalars. A non-zero `shears` or `div` adds the
    linear-elastic terms, which couple the flow components.

    [`flow_adddiag`][fastfields.numpy.flow_adddiag] returns a new array
    instead.

    Parameters
    ----------
    inp : numpy.ndarray
        Accumulator, shape `(*batch, *spatial, ndim)`. Updated in place.
        Must be a writable float32 or float64 array: anything else is
        converted to a copy, and your array is left unchanged.
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
    numpy.ndarray
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
    inp: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Accumulate the flow regulariser's diagonal in place: `inp -= diag(L)`.

    Updates `inp` with `inp -= diag(L)` for the flow regulariser `L`, in one
    kernel. The penalties are scalars. A non-zero `shears` or `div` adds the
    linear-elastic terms, which couple the flow components.

    [`flow_subdiag`][fastfields.numpy.flow_subdiag] returns a new array
    instead.

    Parameters
    ----------
    inp : numpy.ndarray
        Accumulator, shape `(*batch, *spatial, ndim)`. Updated in place.
        Must be a writable float32 or float64 array: anything else is
        converted to a copy, and your array is left unchanged.
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
    numpy.ndarray
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
    mat: np.ndarray,
    vec: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Jacobi preconditioner: solve `(M + diag(L)) x = vec` voxel by voxel.

    Solves `(M + diag(L)) x = vec` voxel by voxel, where `M` is the per-voxel
    matrix `mat` and `diag(L)` is the diagonal of the field regulariser `L`. It
    approximates the inverse of the operator of
    [`field_forward`][fastfields.numpy.field_forward]. Each penalty is given
    per channel and the regulariser never couples channels.

    Batch axes of `mat` and `vec` are broadcast against each other; the spatial
    axes and the last axis must match.

    [`field_precond_`][fastfields.numpy.field_precond_] is the in-place
    version.

    Parameters
    ----------
    mat : numpy.ndarray
        Per-voxel matrix `M` in compact-symmetric packing, shape `(*batch,
        *spatial, C*(C+1)//2)`. Batch axes are broadcast against those of
        `vec`.
    vec : numpy.ndarray
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
    numpy.ndarray
        The solution `x`, shape `(*batch, *spatial, C)` with the broadcast
        batch shape.
    """
    batch, mat, vec = _reg_broadcast(mat, vec, ndim)
    channels = vec.shape[-1]
    out = np.empty(batch + vec.shape[vec.ndim - int(ndim) - 1 :], vec.dtype)
    _ff.field_precond(
        out,
        mat,
        vec,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def field_forward(
    mat: np.ndarray,
    vec: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Apply the regularised system: `(M + L) @ vec`.

    Computes `(M + L) @ vec`, where `M` is the per-voxel matrix `mat` and `L`
    is the field regulariser `L`. This is the operator that
    [`field_relax`][fastfields.numpy.field_relax] solves. Each penalty is given
    per channel and the regulariser never couples channels.

    Batch axes of `mat` and `vec` are broadcast against each other; the spatial
    axes and the last axis must match.

    Parameters
    ----------
    mat : numpy.ndarray
        Per-voxel matrix `M` in compact-symmetric packing, shape `(*batch,
        *spatial, C*(C+1)//2)`. Batch axes are broadcast against those of
        `vec`.
    vec : numpy.ndarray
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
    numpy.ndarray
        `(M + L) @ vec`, shape `(*batch, *spatial, C)` with the broadcast
        batch shape.
    """
    batch, mat, vec = _reg_broadcast(mat, vec, ndim)
    channels = vec.shape[-1]
    out = np.empty(batch + vec.shape[vec.ndim - int(ndim) - 1 :], vec.dtype)
    _ff.field_forward(
        out,
        mat,
        vec,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def field_addmatvec(
    inp: np.ndarray,
    field: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Add the field regulariser applied to a field: `inp + L @ field`.

    Computes `inp + L @ field` for the field regulariser `L`. Each penalty is
    given per channel and the regulariser never couples channels.

    `inp` is copied, never modified;
    [`field_addmatvec_`][fastfields.numpy.field_addmatvec_] is the version that
    updates it in place.

    Parameters
    ----------
    inp : numpy.ndarray
        Array the result is accumulated onto, shape `(*batch, *spatial,
        C)`.
    field : numpy.ndarray
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
    numpy.ndarray
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
    inp: np.ndarray,
    field: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Subtract the field regulariser applied to a field: `inp - L @ field`.

    Computes `inp - L @ field` for the field regulariser `L`. Each penalty is
    given per channel and the regulariser never couples channels.

    `inp` is copied, never modified;
    [`field_submatvec_`][fastfields.numpy.field_submatvec_] is the version that
    updates it in place.

    Parameters
    ----------
    inp : numpy.ndarray
        Array the result is accumulated onto, shape `(*batch, *spatial,
        C)`.
    field : numpy.ndarray
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
    numpy.ndarray
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
    inp: np.ndarray,
    field: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Accumulate the field regulariser in place: `inp += L @ field`.

    Updates `inp` with `inp += L @ field` for the field regulariser `L`, in one
    kernel and without a temporary for the product. Each penalty is given per
    channel and the regulariser never couples channels.

    [`field_addmatvec`][fastfields.numpy.field_addmatvec] is the version that
    returns a new array.

    Parameters
    ----------
    inp : numpy.ndarray
        Accumulator, shape `(*batch, *spatial, C)`. Updated in place. Must
        be a writable float32 or float64 array: anything else is converted
        to a copy, and your array is left unchanged.
    field : numpy.ndarray
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
    numpy.ndarray
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
    inp: np.ndarray,
    field: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Accumulate the field regulariser in place: `inp -= L @ field`.

    Updates `inp` with `inp -= L @ field` for the field regulariser `L`, in one
    kernel and without a temporary for the product. Each penalty is given per
    channel and the regulariser never couples channels.

    [`field_submatvec`][fastfields.numpy.field_submatvec] is the version that
    returns a new array.

    Parameters
    ----------
    inp : numpy.ndarray
        Accumulator, shape `(*batch, *spatial, C)`. Updated in place. Must
        be a writable float32 or float64 array: anything else is converted
        to a copy, and your array is left unchanged.
    field : numpy.ndarray
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
    numpy.ndarray
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
    inp: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Add the diagonal of the field regulariser: `inp + diag(L)`.

    Computes `inp + diag(L)` for the field regulariser `L`. Each penalty is
    given per channel and the regulariser never couples channels.

    `inp` is copied, never modified;
    [`field_adddiag_`][fastfields.numpy.field_adddiag_] updates it in place.

    Parameters
    ----------
    inp : numpy.ndarray
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
    numpy.ndarray
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
    inp: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Subtract the diagonal of the field regulariser: `inp - diag(L)`.

    Computes `inp - diag(L)` for the field regulariser `L`. Each penalty is
    given per channel and the regulariser never couples channels.

    `inp` is copied, never modified;
    [`field_subdiag_`][fastfields.numpy.field_subdiag_] updates it in place.

    Parameters
    ----------
    inp : numpy.ndarray
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
    numpy.ndarray
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
    inp: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Accumulate the field regulariser's diagonal in place: `inp += diag(L)`.

    Updates `inp` with `inp += diag(L)` for the field regulariser `L`, in one
    kernel. Each penalty is given per channel and the regulariser never couples
    channels.

    [`field_adddiag`][fastfields.numpy.field_adddiag] returns a new array
    instead.

    Parameters
    ----------
    inp : numpy.ndarray
        Accumulator, shape `(*batch, *spatial, C)`. Updated in place. Must
        be a writable float32 or float64 array: anything else is converted
        to a copy, and your array is left unchanged.
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
    numpy.ndarray
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
    inp: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Accumulate the field regulariser's diagonal in place: `inp -= diag(L)`.

    Updates `inp` with `inp -= diag(L)` for the field regulariser `L`, in one
    kernel. Each penalty is given per channel and the regulariser never couples
    channels.

    [`field_subdiag`][fastfields.numpy.field_subdiag] returns a new array
    instead.

    Parameters
    ----------
    inp : numpy.ndarray
        Accumulator, shape `(*batch, *spatial, C)`. Updated in place. Must
        be a writable float32 or float64 array: anything else is converted
        to a copy, and your array is left unchanged.
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
    numpy.ndarray
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
# In-place preconditioners and the CG solver
# ---------------------------------------------------------------------------


def field_precond_(
    sol: np.ndarray,
    mat: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Jacobi preconditioner of the regularised system, in place.

    Solves `(M + diag(L)) x = sol` voxel by voxel and stores `x` in `sol`,
    where `M` is the per-voxel matrix `mat` and `diag(L)` is the diagonal of
    the field regulariser `L`. Each penalty is given per channel and the
    regulariser never couples channels.

    Nothing is broadcast: `sol` is written through, so its shape is the shape
    of the answer. [`field_precond`][fastfields.numpy.field_precond] is the
    version that returns a new array.

    Parameters
    ----------
    sol : numpy.ndarray
        The gradient on entry, the preconditioned solution on exit, shape
        `(*batch, *spatial, C)`. Updated in place.
    mat : numpy.ndarray
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
    numpy.ndarray
        `sol` itself, now holding the solution.
    """
    sol = _as_float_array(sol, "sol")
    mat = _as_float_array(mat, "mat")
    channels = sol.shape[-1]
    _ff.field_precond_(
        sol,
        mat,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return sol


def flow_precond_(
    sol: np.ndarray,
    mat: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Jacobi preconditioner of the regularised system, in place.

    Solves `(M + diag(L)) x = sol` voxel by voxel and stores `x` in `sol`,
    where `M` is the per-voxel matrix `mat` and `diag(L)` is the diagonal of
    the flow regulariser `L`. The penalties are scalars. A non-zero `shears` or
    `div` adds the linear-elastic terms, which couple the flow components.

    Nothing is broadcast: `sol` is written through, so its shape is the shape
    of the answer. [`flow_precond`][fastfields.numpy.flow_precond] is the
    version that returns a new array.

    Parameters
    ----------
    sol : numpy.ndarray
        The gradient on entry, the preconditioned solution on exit, shape
        `(*batch, *spatial, ndim)`. Updated in place.
    mat : numpy.ndarray
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
    numpy.ndarray
        `sol` itself, now holding the solution.
    """
    sol = _as_float_array(sol, "sol")
    mat = _as_float_array(mat, "mat")
    _ff.flow_precond_(
        sol,
        mat,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return sol


class CGInfo(NamedTuple):
    """Convergence diagnostics returned by
    [`field_cg`][fastfields.numpy.field_cg].
    """

    #: Number of CG iterations actually run (<= the requested ``nb_iter``).
    iterations: int
    #: Final relative residual ``||g - (H + L) x|| / ||g||``.
    residual: float


def field_cg(
    sol: np.ndarray,
    mat: np.ndarray,
    grd: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
    nb_iter: int = 32,
    tol: float = 1e-8,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, CGInfo]:
    """Solve `(M + L) x = g` by preconditioned conjugate gradients.

    The Krylov alternative to the Gauss-Seidel sweeps of
    [`field_relax`][fastfields.numpy.field_relax] on the same system,
    Jacobi-preconditioned with `(M + diag(L))^-1` as in
    [`field_precond`][fastfields.numpy.field_precond].

    `sol` is both the starting point and the output: it is updated in place and
    returned. Pass zeros for a cold start, or a previous estimate to continue
    from it.

    The iteration stops after `nb_iter` steps, as soon as the relative residual
    falls below `tol` (with `tol <= 0` every iteration is run), or early if the
    operator turns out not to be positive definite along the current search
    direction. Pass `return_info=True` to find out which happened.

    !!! note "CPU only"
        This operation is CPU-only in the underlying library, whose dot
        products need a device-side reduction that the CUDA backend does
        not implement yet. That costs this NumPy wrapper nothing, but it is
        why `fastfields.cupy` has no `field_cg`.

    Parameters
    ----------
    sol : numpy.ndarray
        Starting point on entry, solution on exit, shape `(*batch,
        *spatial, C)`. Updated in place.
    mat : numpy.ndarray
        Per-voxel matrix `M` in compact-symmetric packing, shape `(*batch,
        *spatial, C*(C+1)//2)`.
    grd : numpy.ndarray
        Right-hand side `g`, shape `(*batch, *spatial, C)`.
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
    nb_iter : int, default=32
        Maximum number of iterations.
    tol : float, default=1e-08
        Relative residual at which to stop; zero or negative disables the
        early exit.
    return_info : bool, default=False
        Also return a [`CGInfo`][fastfields.numpy.CGInfo] with the
        iteration count and final residual.

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, CGInfo)
        `sol`, updated in place. With `return_info=True`, a tuple of `sol`
        and a [`CGInfo`][fastfields.numpy.CGInfo].
    """
    sol = _as_float_array(sol, "sol")
    mat = _as_float_array(mat, "mat")
    grd = _as_float_array(grd, "grd")
    channels = sol.shape[-1]
    iterations, residual = _ff.field_cg(
        sol,
        mat,
        grd,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
        nb_iter=int(nb_iter),
        tol=float(tol),
    )
    if return_info:
        return sol, CGInfo(int(iterations), float(residual))
    return sol


# ---------------------------------------------------------------------------
# Stencil (kernel) accumulate variants
# ---------------------------------------------------------------------------
#
# The same `inp ± op(...)` pattern as the *_adddiag / *_subdiag family, for
# the materialised Toeplitz stencil rather than the diagonal. `inp` is shaped
# like the output of `field_kernel` / `flow_kernel`.


def field_addkernel(
    inp: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Add the stencil of the field regulariser: `inp + K`.

    Computes `inp + K`, where `K` is the stencil that
    [`field_kernel`][fastfields.numpy.field_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil and is copied, never
    modified; [`field_addkernel_`][fastfields.numpy.field_addkernel_] updates
    it in place.

    Parameters
    ----------
    inp : numpy.ndarray
        Array the stencil is accumulated onto, shaped like the output of
        [`field_kernel`][fastfields.numpy.field_kernel].
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
    numpy.ndarray
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
    inp: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Subtract the stencil of the field regulariser: `inp - K`.

    Computes `inp - K`, where `K` is the stencil that
    [`field_kernel`][fastfields.numpy.field_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil and is copied, never
    modified; [`field_subkernel_`][fastfields.numpy.field_subkernel_] updates
    it in place.

    Parameters
    ----------
    inp : numpy.ndarray
        Array the stencil is accumulated onto, shaped like the output of
        [`field_kernel`][fastfields.numpy.field_kernel].
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
    numpy.ndarray
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
    inp: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Accumulate the field regulariser's stencil in place: `inp += K`.

    Updates `inp` with `inp += K`, where `K` is the stencil that
    [`field_kernel`][fastfields.numpy.field_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil.
    [`field_addkernel`][fastfields.numpy.field_addkernel] returns a new array
    instead.

    Parameters
    ----------
    inp : numpy.ndarray
        Accumulator, shaped like the output of
        [`field_kernel`][fastfields.numpy.field_kernel]. Updated in place.
        Must be a writable float32 or float64 array: anything else is
        converted to a copy, and your array is left unchanged.
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
    numpy.ndarray
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
    inp: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Accumulate the field regulariser's stencil in place: `inp -= K`.

    Updates `inp` with `inp -= K`, where `K` is the stencil that
    [`field_kernel`][fastfields.numpy.field_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil.
    [`field_subkernel`][fastfields.numpy.field_subkernel] returns a new array
    instead.

    Parameters
    ----------
    inp : numpy.ndarray
        Accumulator, shaped like the output of
        [`field_kernel`][fastfields.numpy.field_kernel]. Updated in place.
        Must be a writable float32 or float64 array: anything else is
        converted to a copy, and your array is left unchanged.
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
    numpy.ndarray
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
    inp: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Add the stencil of the flow regulariser: `inp + K`.

    Computes `inp + K`, where `K` is the stencil that
    [`flow_kernel`][fastfields.numpy.flow_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil and is copied, never
    modified; [`flow_addkernel_`][fastfields.numpy.flow_addkernel_] updates it
    in place.

    Parameters
    ----------
    inp : numpy.ndarray
        Array the stencil is accumulated onto, shaped like the output of
        [`flow_kernel`][fastfields.numpy.flow_kernel].
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
    numpy.ndarray
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
    inp: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Subtract the stencil of the flow regulariser: `inp - K`.

    Computes `inp - K`, where `K` is the stencil that
    [`flow_kernel`][fastfields.numpy.flow_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil and is copied, never
    modified; [`flow_subkernel_`][fastfields.numpy.flow_subkernel_] updates it
    in place.

    Parameters
    ----------
    inp : numpy.ndarray
        Array the stencil is accumulated onto, shaped like the output of
        [`flow_kernel`][fastfields.numpy.flow_kernel].
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
    numpy.ndarray
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
    inp: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Accumulate the flow regulariser's stencil in place: `inp += K`.

    Updates `inp` with `inp += K`, where `K` is the stencil that
    [`flow_kernel`][fastfields.numpy.flow_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil.
    [`flow_addkernel`][fastfields.numpy.flow_addkernel] returns a new array
    instead.

    Parameters
    ----------
    inp : numpy.ndarray
        Accumulator, shaped like the output of
        [`flow_kernel`][fastfields.numpy.flow_kernel]. Updated in place.
        Must be a writable float32 or float64 array: anything else is
        converted to a copy, and your array is left unchanged.
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
    numpy.ndarray
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
    inp: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Accumulate the flow regulariser's stencil in place: `inp -= K`.

    Updates `inp` with `inp -= K`, where `K` is the stencil that
    [`flow_kernel`][fastfields.numpy.flow_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil.
    [`flow_subkernel`][fastfields.numpy.flow_subkernel] returns a new array
    instead.

    Parameters
    ----------
    inp : numpy.ndarray
        Accumulator, shaped like the output of
        [`flow_kernel`][fastfields.numpy.flow_kernel]. Updated in place.
        Must be a writable float32 or float64 array: anything else is
        converted to a copy, and your array is left unchanged.
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
    numpy.ndarray
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
