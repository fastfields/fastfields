"""Compact-symmetric matrix operations for NumPy.

A matrix with `C` channels is stored along the last axis as `C*(C+1)//2`
values: the diagonal first, then the rows of the upper triangle. `mat`
therefore has a last axis of length `C*(C+1)//2` and vectors one of length
`C`.
"""

from __future__ import annotations

import math

import fastfields.dlpack as _ff

import numpy as np

from ._util import (
    _as_float_array,
    _bcast_view,
    _broadcast_batch,
    _validate_inplace,
)

__all__ = [
    "sym_channels_from_packed",
    "sym_matvec",
    "sym_matvec_backward",
    "sym_addmatvec_",
    "sym_submatvec_",
    "sym_solve",
    "sym_solve_",
    "sym_invert",
    "sym_invert_",
]


def sym_channels_from_packed(packed_len: int) -> int:
    """Channel count `C` of a compact-symmetric matrix from its packed length.

    Inverts `packed_len = C*(C+1)//2`.

    Parameters
    ----------
    packed_len : int
        Length of the packed last axis.

    Returns
    -------
    int
        The channel count `C`.

    Raises
    ------
    ValueError
        If `packed_len` is not a triangular number.
    """
    c = int((math.isqrt(8 * packed_len + 1) - 1) // 2)
    if c * (c + 1) // 2 != packed_len:
        raise ValueError(
            f"packed length {packed_len} is not a triangular number "
            "(expected C*(C+1)/2 for some integer C)"
        )
    return c


def _check_sym(
    mat: np.ndarray,
    vec: np.ndarray,
    matname: str = "mat",
    vecname: str = "vec",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate dtypes/channels of a packed matrix + vector (no batch check).

    Batch dims are broadcast later, so we only enforce the channel relation
    ``vec.shape[-1] == C``.

    !!! note
        The binding dispatches a single ``scalar_t`` for both operands, so
        ``mat`` and ``vec`` must share a dtype. Rather than silently casting
        ``vec`` to ``mat``'s dtype (which would drop precision for a float64
        ``vec`` against a float32 ``mat``), both are promoted to
        ``numpy.result_type(mat, vec)`` -- the lower-precision operand is
        upcast, never the reverse.

    Parameters
    ----------
    mat : numpy.ndarray
        Packed compact-symmetric matrix, trailing dim ``C*(C+1)//2``.
    vec : numpy.ndarray
        Vector, trailing dim ``C``.
    matname, vecname : str, optional
        Argument names used in error messages.

    Returns
    -------
    mat : numpy.ndarray
        Float matrix, promoted to the common dtype.
    vec : numpy.ndarray
        Float vector, promoted to the common dtype.
    c : int
        The channel count ``C``.

    Raises
    ------
    ValueError
        If ``vec``'s channel count does not match the matrix packing.
    """
    mat = _as_float_array(mat, matname)
    vec = _as_float_array(vec, vecname)
    dtype = np.result_type(mat.dtype, vec.dtype)
    mat = mat.astype(dtype, copy=False)
    vec = vec.astype(dtype, copy=False)
    c = sym_channels_from_packed(mat.shape[-1])
    if vec.shape[-1] != c:
        raise ValueError(
            f"{vecname} has {vec.shape[-1]} channels but the packed matrix "
            f"encodes {c} channels"
        )
    return mat, vec, c


def sym_matvec(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Product of compact-symmetric matrices with vectors.

    A compact-symmetric matrix with `C` channels is stored along the last axis
    as `C*(C+1)//2` values: the diagonal first, then the rows of the upper
    triangle. Vectors have a last axis of length `C`.

    Batch axes of `mat` and `vec` are broadcast against each other.

    Parameters
    ----------
    mat : numpy.ndarray
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`.
    vec : numpy.ndarray
        Vectors, shape `(*B, C)`.

    Returns
    -------
    numpy.ndarray
        `mat @ vec`, shape `(*B, C)` with the broadcast batch shape.
    """
    mat, vec, c = _check_sym(mat, vec)
    batch, (mat_b, vec_b) = _broadcast_batch([(mat, 1), (vec, 1)])
    out = np.empty(batch + (c,), dtype=mat.dtype)
    _ff.sym_matvec(out, mat_b, vec_b)
    return out


def _addsub_matvec_(
    out0: np.ndarray,
    mat: np.ndarray,
    vec: np.ndarray,
    binding,
) -> np.ndarray:
    """Shared in-place ``out0 (+|-)= H @ vec`` implementation.

    ``out0`` is mutated in place and returned. It fixes the batch (leading)
    shape; ``mat`` and ``vec`` are broadcast (zero-copy) to that batch shape.
    The trailing-``_`` contract requires the caller's ``out0`` buffer to be
    updated, so we write through it via DLPack and never materialise a private
    copy (matching the cupy backend).
    """
    # out0 fixes the output buffer: validate it in place (no copy) so the
    # binding writes through the caller's array.
    out0 = _validate_inplace(out0, "out0")
    # mat/vec must match out0's dtype (its buffer cannot be re-typed in place).
    mat = _as_float_array(mat, "mat", dtype=out0.dtype)
    vec = _as_float_array(vec, "vec", dtype=out0.dtype)
    c = sym_channels_from_packed(mat.shape[-1])
    if vec.shape[-1] != c:
        raise ValueError(
            f"vec has {vec.shape[-1]} channels but the packed matrix "
            f"encodes {c} channels"
        )
    if out0.shape[-1] != c:
        raise ValueError("out0 must have the same channel count as vec")
    # out0 fixes the batch; broadcast mat/vec onto it (zero-copy views).
    batch = out0.shape[:-1]
    mat_b = _bcast_view(mat, batch + (mat.shape[-1],))
    vec_b = _bcast_view(vec, batch + (c,))
    binding(out0, mat_b, vec_b)
    return out0


def sym_addmatvec_(
    out0: np.ndarray, mat: np.ndarray, vec: np.ndarray
) -> np.ndarray:
    """Accumulate a matrix-vector product in place: `out0 += mat @ vec`.

    `out0` fixes the batch shape; `mat` and `vec` are broadcast to it. Note
    that the accumulator comes first, unlike in
    [`sym_matvec`][fastfields.numpy.sym_matvec].

    Parameters
    ----------
    out0 : numpy.ndarray
        Accumulator, shape `(*B, C)`. Updated in place. It fixes the batch
        shape of the call.
    mat : numpy.ndarray
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`, broadcast to
        the batch shape of `out0`.
    vec : numpy.ndarray
        Vectors, shape `(*B, C)`, broadcast to the batch shape of `out0`.

    Returns
    -------
    numpy.ndarray
        `out0` itself, now holding `out0 + mat @ vec`.

    Raises
    ------
    TypeError
        If `out0` is not a float32 or float64 NumPy array.
    ValueError
        If the channel counts disagree, or `mat` or `vec` cannot be
        broadcast to the batch shape of `out0`.
    """
    return _addsub_matvec_(out0, mat, vec, _ff.sym_addmatvec_)


def sym_submatvec_(
    out0: np.ndarray, mat: np.ndarray, vec: np.ndarray
) -> np.ndarray:
    """Subtract a matrix-vector product in place: `out0 -= mat @ vec`.

    The subtracting twin of
    [`sym_addmatvec_`][fastfields.numpy.sym_addmatvec_], with the same
    contract.

    Parameters
    ----------
    out0 : numpy.ndarray
        Accumulator, shape `(*B, C)`. Updated in place. It fixes the batch
        shape of the call.
    mat : numpy.ndarray
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`, broadcast to
        the batch shape of `out0`.
    vec : numpy.ndarray
        Vectors, shape `(*B, C)`, broadcast to the batch shape of `out0`.

    Returns
    -------
    numpy.ndarray
        `out0` itself, now holding `out0 - mat @ vec`.

    Raises
    ------
    TypeError
        If `out0` is not a float32 or float64 NumPy array.
    ValueError
        If the channel counts disagree, or `mat` or `vec` cannot be
        broadcast to the batch shape of `out0`.
    """
    return _addsub_matvec_(out0, mat, vec, _ff.sym_submatvec_)


def sym_matvec_backward(grad: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Gradient of `sym_matvec` with respect to the matrix.

    Given the upstream gradient of `mat @ vec` and the `vec` of the forward
    pass, returns the gradient with respect to the packed matrix, in the same
    packing. Batch axes are broadcast against each other.

    Only numpy and cupy expose this function. On torch it is the adjoint behind
    `sym_matvec`'s autograd support, reached by calling `.backward()`.

    Parameters
    ----------
    grad : numpy.ndarray
        Upstream gradient, shape `(*B, C)`.
    vec : numpy.ndarray
        The vector of the forward pass, shape `(*B, C)`.

    Returns
    -------
    numpy.ndarray
        Gradient with respect to `mat`, shape `(*B, C*(C+1)//2)`.

    Raises
    ------
    ValueError
        If `grad` and `vec` have different channel counts.
    """
    grad = _as_float_array(grad, "grad")
    vec = _as_float_array(vec, "vec", dtype=grad.dtype)
    if grad.shape[-1] != vec.shape[-1]:
        raise ValueError("grad and vec must have the same channel count")
    c = grad.shape[-1]
    packed = c * (c + 1) // 2
    batch, (grad_b, vec_b) = _broadcast_batch([(grad, 1), (vec, 1)])
    out = np.empty(batch + (packed,), dtype=grad.dtype)
    _ff.sym_matvec_backward(out, grad_b, vec_b)
    return out


def sym_solve(
    mat: np.ndarray, vec: np.ndarray, weight: np.ndarray | None = None
) -> np.ndarray:
    """Solve compact-symmetric systems, optionally with a diagonal added.

    A compact-symmetric matrix with `C` channels is stored along the last axis
    as `C*(C+1)//2` values: the diagonal first, then the rows of the upper
    triangle. Vectors have a last axis of length `C`. Solves `(mat +
    diag(weight)) x = vec` for `x`, voxel by voxel. Batch axes of all inputs
    are broadcast against each other.

    [`sym_solve_`][fastfields.numpy.sym_solve_] is the in-place version.

    Parameters
    ----------
    mat : numpy.ndarray
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`.
    vec : numpy.ndarray
        Right-hand sides, shape `(*B, C)`.
    weight : numpy.ndarray, optional
        Diagonal added to `mat` before solving, shape `(*B, C)`.

    Returns
    -------
    numpy.ndarray
        The solution `x`, shape `(*B, C)` with the broadcast batch shape.

    Raises
    ------
    ValueError
        If the channel counts of `mat`, `vec` and `weight` disagree.
    """
    mat, vec, c = _check_sym(mat, vec)
    if weight is None:
        batch, (mat_b, vec_b) = _broadcast_batch([(mat, 1), (vec, 1)])
        out = np.empty(batch + (c,), dtype=mat.dtype)
        _ff.sym_solve(out, mat_b, vec_b)
    else:
        w = _as_float_array(weight, "weight", dtype=mat.dtype)
        if w.shape[-1] != c:
            raise ValueError("weight must have the same channel count as vec")
        batch, (mat_b, vec_b, w_b) = _broadcast_batch(
            [(mat, 1), (vec, 1), (w, 1)]
        )
        out = np.empty(batch + (c,), dtype=mat.dtype)
        _ff.sym_solve(out, mat_b, vec_b, w_b)
    return out


def sym_invert(mat: np.ndarray) -> np.ndarray:
    """Invert compact-symmetric matrices.

    A compact-symmetric matrix with `C` channels is stored along the last axis
    as `C*(C+1)//2` values: the diagonal first, then the rows of the upper
    triangle. Vectors have a last axis of length `C`. The inverse is symmetric
    too and is returned in the same packing.

    [`sym_invert_`][fastfields.numpy.sym_invert_] is the in-place version.

    Parameters
    ----------
    mat : numpy.ndarray
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`.

    Returns
    -------
    numpy.ndarray
        The packed inverses, with the shape of `mat`.
    """
    mat = _as_float_array(mat, "mat")
    # Output must be a real C-contiguous buffer even if ``mat`` is strided.
    out = np.empty(mat.shape, dtype=mat.dtype)
    _ff.sym_invert(out, mat)
    return out


def sym_solve_(
    inp_out: np.ndarray, mat: np.ndarray, weight: np.ndarray | None = None
) -> np.ndarray:
    """Solve compact-symmetric linear systems in place.

    Solves `(mat + diag(weight)) x = inp_out` for `x` and stores `x` in
    `inp_out`. `inp_out` fixes the batch shape; `mat` and `weight` are
    broadcast to it. Note that the right-hand side comes first, unlike in
    [`sym_solve`][fastfields.numpy.sym_solve].

    Parameters
    ----------
    inp_out : numpy.ndarray
        Right-hand sides, shape `(*B, C)`, on entry; the solutions on exit.
    mat : numpy.ndarray
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`, broadcast to
        the batch shape of `inp_out`.
    weight : numpy.ndarray, optional
        Diagonal added to `mat`, shape `(*B, C)`, broadcast to the batch
        shape of `inp_out`.

    Returns
    -------
    numpy.ndarray
        `inp_out` itself, now holding the solution.

    Raises
    ------
    TypeError
        If `inp_out` is not a float32 or float64 NumPy array.
    ValueError
        If the channel counts disagree, or `mat` or `weight` cannot be
        broadcast to the batch shape of `inp_out`.
    """
    inp_out = _validate_inplace(inp_out, "inp_out")
    mat = _as_float_array(mat, "mat", dtype=inp_out.dtype)
    c = sym_channels_from_packed(mat.shape[-1])
    if inp_out.shape[-1] != c:
        raise ValueError(
            f"inp_out has {inp_out.shape[-1]} channels but the packed "
            f"matrix encodes {c} channels"
        )
    batch = inp_out.shape[:-1]
    mat_b = _bcast_view(mat, batch + (mat.shape[-1],))
    if weight is None:
        _ff.sym_solve_(inp_out, mat_b, None)
    else:
        weight = _as_float_array(weight, "weight", dtype=inp_out.dtype)
        if weight.shape[-1] != c:
            raise ValueError(
                f"weight has {weight.shape[-1]} channels but the packed "
                f"matrix encodes {c} channels"
            )
        weight_b = _bcast_view(weight, batch + (c,))
        _ff.sym_solve_(inp_out, mat_b, weight_b)
    return inp_out


def sym_invert_(mat: np.ndarray) -> np.ndarray:
    """Invert compact-symmetric matrices in place.

    The inverse is symmetric too and replaces `mat` in the same packing.

    Parameters
    ----------
    mat : numpy.ndarray
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`, on entry;
        their inverses on exit.

    Returns
    -------
    numpy.ndarray
        `mat` itself, now holding the packed inverses.

    Raises
    ------
    TypeError
        If `mat` is not a float32 or float64 NumPy array.
    """
    _validate_inplace(mat, "mat")
    _ff.sym_invert_(mat)
    return mat
