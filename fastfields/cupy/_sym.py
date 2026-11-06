"""Compact-symmetric matrix operations (cupy).

The Hessian ``H`` is stored in a compact, diagonal-then-rows packing. For a
per-batch ``C x C`` symmetric matrix the packed length is ``C*(C+1)/2``:

* ``C == 2`` -> ``[h00, h11, h01]``
* ``C == 3`` -> ``[h00, h11, h22, h01, h02, h12]``

``vec`` / ``out`` vectors are shaped ``(..., C)``.
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
    """Return ``out = H @ vec`` (functional).

    Batch (leading) dims of ``mat`` and ``vec`` are broadcast (zero-copy);
    ``out`` has the broadcast batch shape + ``vec``'s core dim ``(C,)``.
    """
    cp = cupy()
    mat = as_gpu_array(mat, name="mat")
    vec = as_gpu_array(vec, name="vec")
    batch, (mat_b, vec_b) = broadcast_batch([(mat, 1), (vec, 1)])
    out = cp.empty(batch + (vec.shape[-1],), dtype=vec.dtype)
    _ff.sym_matvec(out, mat_b, vec_b, current_stream_ptr())
    return out


def sym_matvec_backward(grad: Any, vec: Any) -> Any:
    """Backward of [`sym_matvec`][fastfields.cupy.sym_matvec] w.r.t. the matrix.

    ``grad`` and ``vec`` are ``(..., C)`` (batch dims broadcast); returns the
    gradient in the same compact-symmetric packing, shape ``(..., C*(C+1)/2)``.
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
    """In-place accumulate: ``out0 += H @ vec``. Returns ``out0``.

    ``out0`` fixes the batch shape; ``mat`` and ``vec`` are broadcast to it.
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
    """In-place accumulate: ``out0 -= H @ vec``. Returns ``out0``.

    ``out0`` fixes the batch shape; ``mat`` and ``vec`` are broadcast to it.
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
    """Return ``out = (H + diag(weight)) \\ vec`` (functional).

    ``weight`` is optional. Batch dims of ``mat``/``vec``/``weight`` are
    broadcast (zero-copy).
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
    """In-place solve ``inp_out = (H + diag(weight)) \\ inp_out``.

    Returns ``inp_out``. It fixes the batch shape; ``mat``/``weight`` are
    broadcast to it.
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
    """Return ``inv(H)`` (functional); both compact-symmetric."""
    cp = cupy()
    mat = as_gpu_array(mat, name="mat")
    out = cp.empty_like(mat)
    _ff.sym_invert(out, mat, current_stream_ptr())
    return out


def sym_invert_(mat: Any) -> Any:
    """In-place invert: ``mat = inv(mat)``. Returns ``mat``."""
    mat = require_gpu_writethrough(mat, name="mat")
    _ff.sym_invert_(mat, current_stream_ptr())
    return mat
