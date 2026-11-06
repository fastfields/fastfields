"""Compact-symmetric matrix operations for CuPy.

A matrix with `C` channels is stored as `C*(C+1)//2` values along the last
axis, the diagonal first and then the rows of the upper triangle. For example
`C == 2` is `[h00, h11, h01]` and `C == 3` is
`[h00, h11, h22, h01, h02, h12]`. Vectors have a last axis of length `C`.
"""

from __future__ import annotations

from typing import Any

import fastfields.dlpack as _ff

from ._util import (
    as_gpu_array,
    broadcast_batch,
    broadcast_to_batch,
    cupy,
    current_stream_ptr,
    require_gpu_writethrough,
)

__all__ = [
    "sym_matvec",
    "sym_matvec_backward",
    "sym_addmatvec_",
    "sym_submatvec_",
    "sym_solve",
    "sym_solve_",
    "sym_invert",
    "sym_invert_",
]


def sym_matvec(mat: Any, vec: Any) -> Any:
    """Product of compact-symmetric matrices with vectors.

    A compact-symmetric matrix with `C` channels is stored along the last axis
    as `C*(C+1)//2` values: the diagonal first, then the rows of the upper
    triangle. Vectors have a last axis of length `C`.

    Batch axes of `mat` and `vec` are broadcast against each other.

    Parameters
    ----------
    mat : cupy.ndarray
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`.
    vec : cupy.ndarray
        Vectors, shape `(*B, C)`.

    Returns
    -------
    cupy.ndarray
        `mat @ vec`, shape `(*B, C)` with the broadcast batch shape.
    """
    cp = cupy()
    mat = as_gpu_array(mat, name="mat")
    vec = as_gpu_array(vec, name="vec")
    batch, (mat_b, vec_b) = broadcast_batch([(mat, 1), (vec, 1)])
    out = cp.empty(batch + (vec.shape[-1],), dtype=vec.dtype)
    _ff.sym_matvec(out, mat_b, vec_b, current_stream_ptr())
    return out


def sym_matvec_backward(grad: Any, vec: Any) -> Any:
    """Gradient of `sym_matvec` with respect to the matrix.

    Given the upstream gradient of `mat @ vec` and the `vec` of the forward
    pass, returns the gradient with respect to the packed matrix, in the same
    packing. Batch axes are broadcast against each other.

    Only numpy and cupy expose this function. On torch it is the adjoint behind
    `sym_matvec`'s autograd support, reached by calling `.backward()`.

    Parameters
    ----------
    grad : cupy.ndarray
        Upstream gradient, shape `(*B, C)`.
    vec : cupy.ndarray
        The vector of the forward pass, shape `(*B, C)`.

    Returns
    -------
    cupy.ndarray
        Gradient with respect to `mat`, shape `(*B, C*(C+1)//2)`.
    """
    cp = cupy()
    grad = as_gpu_array(grad, name="grad")
    vec = as_gpu_array(vec, name="vec")
    c = vec.shape[-1]
    packed = c * (c + 1) // 2
    batch, (grad_b, vec_b) = broadcast_batch([(grad, 1), (vec, 1)])
    out = cp.empty(batch + (packed,), dtype=vec.dtype)
    _ff.sym_matvec_backward(out, grad_b, vec_b, current_stream_ptr())
    return out


def sym_addmatvec_(out0: Any, mat: Any, vec: Any) -> Any:
    """Accumulate a matrix-vector product in place: `out0 += mat @ vec`.

    `out0` fixes the batch shape; `mat` and `vec` are broadcast to it. Note
    that the accumulator comes first, unlike in
    [`sym_matvec`][fastfields.cupy.sym_matvec].

    Parameters
    ----------
    out0 : cupy.ndarray
        Accumulator, shape `(*B, C)`. Updated in place. It fixes the batch
        shape of the call.
    mat : cupy.ndarray
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`, broadcast to
        the batch shape of `out0`.
    vec : cupy.ndarray
        Vectors, shape `(*B, C)`, broadcast to the batch shape of `out0`.

    Returns
    -------
    cupy.ndarray
        `out0` itself, now holding `out0 + mat @ vec`.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    out0 = require_gpu_writethrough(out0, name="out0")
    mat = as_gpu_array(mat, name="mat")
    vec = as_gpu_array(vec, name="vec")
    batch = out0.shape[:-1]
    mat_b = broadcast_to_batch(mat, batch, 1)
    vec_b = broadcast_to_batch(vec, batch, 1)
    _ff.sym_addmatvec_(out0, mat_b, vec_b, current_stream_ptr())
    return out0


def sym_submatvec_(out0: Any, mat: Any, vec: Any) -> Any:
    """Subtract a matrix-vector product in place: `out0 -= mat @ vec`.

    The subtracting twin of [`sym_addmatvec_`][fastfields.cupy.sym_addmatvec_],
    with the same contract.

    Parameters
    ----------
    out0 : cupy.ndarray
        Accumulator, shape `(*B, C)`. Updated in place. It fixes the batch
        shape of the call.
    mat : cupy.ndarray
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`, broadcast to
        the batch shape of `out0`.
    vec : cupy.ndarray
        Vectors, shape `(*B, C)`, broadcast to the batch shape of `out0`.

    Returns
    -------
    cupy.ndarray
        `out0` itself, now holding `out0 - mat @ vec`.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    out0 = require_gpu_writethrough(out0, name="out0")
    mat = as_gpu_array(mat, name="mat")
    vec = as_gpu_array(vec, name="vec")
    batch = out0.shape[:-1]
    mat_b = broadcast_to_batch(mat, batch, 1)
    vec_b = broadcast_to_batch(vec, batch, 1)
    _ff.sym_submatvec_(out0, mat_b, vec_b, current_stream_ptr())
    return out0


def sym_solve(mat: Any, vec: Any, weight: Any = None) -> Any:
    """Solve compact-symmetric systems, optionally with a diagonal added.

    A compact-symmetric matrix with `C` channels is stored along the last axis
    as `C*(C+1)//2` values: the diagonal first, then the rows of the upper
    triangle. Vectors have a last axis of length `C`. Solves `(mat +
    diag(weight)) x = vec` for `x`, voxel by voxel. Batch axes of all inputs
    are broadcast against each other.

    [`sym_solve_`][fastfields.cupy.sym_solve_] is the in-place version.

    Parameters
    ----------
    mat : cupy.ndarray
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`.
    vec : cupy.ndarray
        Right-hand sides, shape `(*B, C)`.
    weight : cupy.ndarray, optional
        Diagonal added to `mat` before solving, shape `(*B, C)`.

    Returns
    -------
    cupy.ndarray
        The solution `x`, shape `(*B, C)` with the broadcast batch shape.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    cp = cupy()
    mat = as_gpu_array(mat, name="mat")
    vec = as_gpu_array(vec, name="vec")
    if weight is None:
        batch, (mat_b, vec_b) = broadcast_batch([(mat, 1), (vec, 1)])
        out = cp.empty(batch + (vec.shape[-1],), dtype=vec.dtype)
        _ff.sym_solve(out, mat_b, vec_b, None, current_stream_ptr())
    else:
        weight = as_gpu_array(weight, name="weight")
        batch, (mat_b, vec_b, weight_b) = broadcast_batch(
            [(mat, 1), (vec, 1), (weight, 1)]
        )
        out = cp.empty(batch + (vec.shape[-1],), dtype=vec.dtype)
        _ff.sym_solve(out, mat_b, vec_b, weight_b, current_stream_ptr())
    return out


def sym_solve_(inp_out: Any, mat: Any, weight: Any = None) -> Any:
    """Solve compact-symmetric linear systems in place.

    Solves `(mat + diag(weight)) x = inp_out` for `x` and stores `x` in
    `inp_out`. `inp_out` fixes the batch shape; `mat` and `weight` are
    broadcast to it. Note that the right-hand side comes first, unlike in
    [`sym_solve`][fastfields.cupy.sym_solve].

    Parameters
    ----------
    inp_out : cupy.ndarray
        Right-hand sides, shape `(*B, C)`, on entry; the solutions on exit.
    mat : cupy.ndarray
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`, broadcast to
        the batch shape of `inp_out`.
    weight : cupy.ndarray, optional
        Diagonal added to `mat`, shape `(*B, C)`, broadcast to the batch
        shape of `inp_out`.

    Returns
    -------
    cupy.ndarray
        `inp_out` itself, now holding the solution.

    Raises
    ------
    TypeError
        If an argument is not a float32 or float64 CuPy array.
    """
    inp_out = require_gpu_writethrough(inp_out, name="inp_out")
    mat = as_gpu_array(mat, name="mat")
    batch = inp_out.shape[:-1]
    mat_b = broadcast_to_batch(mat, batch, 1)
    if weight is not None:
        weight = as_gpu_array(weight, name="weight")
        weight = broadcast_to_batch(weight, batch, 1)
    _ff.sym_solve_(inp_out, mat_b, weight, current_stream_ptr())
    return inp_out


def sym_invert(mat: Any) -> Any:
    """Invert compact-symmetric matrices.

    A compact-symmetric matrix with `C` channels is stored along the last axis
    as `C*(C+1)//2` values: the diagonal first, then the rows of the upper
    triangle. Vectors have a last axis of length `C`. The inverse is symmetric
    too and is returned in the same packing.

    [`sym_invert_`][fastfields.cupy.sym_invert_] is the in-place version.

    Parameters
    ----------
    mat : cupy.ndarray
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`.

    Returns
    -------
    cupy.ndarray
        The packed inverses, with the shape of `mat`.

    Raises
    ------
    TypeError
        If `mat` is not a float32 or float64 CuPy array.
    """
    cp = cupy()
    mat = as_gpu_array(mat, name="mat")
    out = cp.empty_like(mat)
    _ff.sym_invert(out, mat, current_stream_ptr())
    return out


def sym_invert_(mat: Any) -> Any:
    """Invert compact-symmetric matrices in place.

    The inverse is symmetric too and replaces `mat` in the same packing.

    Parameters
    ----------
    mat : cupy.ndarray
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`, on entry;
        their inverses on exit.

    Returns
    -------
    cupy.ndarray
        `mat` itself, now holding the packed inverses.

    Raises
    ------
    TypeError
        If `mat` is not a float32 or float64 CuPy array.
    """
    mat = require_gpu_writethrough(mat, name="mat")
    _ff.sym_invert_(mat, current_stream_ptr())
    return mat
