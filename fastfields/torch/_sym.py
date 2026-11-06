"""Linear algebra on batches of compact-symmetric matrices, with autograd.

A compact-symmetric matrix of channel size ``C`` is stored along the last axis
as ``C*(C+1)//2`` values: the diagonal first, then the rows of the upper
triangle, e.g. for ``C == 3``::

    [ a d e ]
    [ . b f ]  =>  [a b c d e f]
    [ . . c ]

These wrappers mirror the autograd structure of ``jitfields/jitfields/sym.py``
(classes ``MatVec`` and ``Solve``).

In-place (``_``-suffixed) forms mirror the numpy/cupy convention
(``sym_solve_(inp_out, mat, weight=None)``, ``sym_invert_(mat)`` -- note the
argument order matches numpy/cupy exactly, which is *not* the same order as
the out-of-place ``sym_solve(mat, vec, weight=None)``). Whether an in-place
form is differentiable follows the per-op rule in ``API_CONTRACT.md``
("In-place policy"):

* [`sym_solve_`][fastfields.torch.sym_solve_] **is** differentiable (w.r.t. the
  mutated right-hand side): the backward of
  [`sym_solve`][fastfields.torch.sym_solve] never reads the pre-mutation
  right-hand side, only the saved ``mat``/``weight``, so overwriting it in
  place destroys no information backward needs.
* [`sym_invert_`][fastfields.torch.sym_invert_] is **not** differentiable --
  for the same reason as [`sym_invert`][fastfields.torch.sym_invert]: no
  gradient is implemented for the packed inverse on this backend at all,
  mirroring ``jitfields`` (which rejects a grad-requiring ``mat`` outright).
  This is an *implementation gap*, not a mathematical obstruction: the matrix
  inverse is perfectly differentiable (``dL/dA = -B (dL/dB) B`` for ``B =
  A^-1``, symmetric), and note that formula needs only the **inverse** -- i.e.
  the value this op leaves behind -- so even the in-place form could support a
  gradient if one were implemented. Calling ``.backward()`` through either
  form's output raises ``RuntimeError``.
"""

from __future__ import annotations

import math
from typing import Optional

import fastfields.dlpack as _fb

import torch
from torch import Tensor

from ._util import (
    check_dtype,
    check_inplace_leaf,
    raise_not_differentiable,
    stream_ptr,
)

__all__ = [
    "sym_matvec",
    "sym_addmatvec_",
    "sym_submatvec_",
    "sym_solve",
    "sym_solve_",
    "sym_invert",
    "sym_invert_",
]


def _channels_from_packed(packed_len: int) -> int:
    """Return the channel count ``C`` with ``C*(C+1)//2 == packed_len``.

    Raises
    ------
    ValueError
        If ``packed_len`` is not a triangular number ``C*(C+1)/2``.
    """
    c = int((math.isqrt(8 * packed_len + 1) - 1) // 2)
    if c * (c + 1) // 2 != packed_len:
        raise ValueError(
            f"packed length {packed_len} is not a triangular number "
            "(expected C*(C+1)/2 for some integer C)"
        )
    return c


def _check_sym(mat: Tensor, vec: Tensor) -> int:
    """Validate a packed matrix / vector pair and return the channel count.

    Verifies that ``mat``'s packed trailing dim ``C*(C+1)//2`` matches the
    channel count ``C`` implied by ``vec``'s trailing dim, so a mismatched pair
    cannot reach the raw binding (which would OOB-read / segfault). Batch dims
    are broadcast by the caller, so only the channel relation is enforced here.

    Parameters
    ----------
    mat : `(..., C*(C+1)//2) tensor`
        Packed compact-symmetric matrix.
    vec : `(..., C) tensor`
        Vector.

    Returns
    -------
    int
        The channel count ``C``.

    Raises
    ------
    ValueError
        If ``vec``'s channel count does not match the matrix packing.
    """
    c = _channels_from_packed(mat.shape[-1])
    if vec.shape[-1] != c:
        raise ValueError(
            f"vec has {vec.shape[-1]} channels but the packed matrix "
            f"encodes {c} channels (packed length {mat.shape[-1]})"
        )
    return c


def sym_matvec(mat: Tensor, vec: Tensor) -> Tensor:
    """Product of compact-symmetric matrices with vectors.

    A compact-symmetric matrix with `C` channels is stored along the last axis
    as `C*(C+1)//2` values: the diagonal first, then the rows of the upper
    triangle. Vectors have a last axis of length `C`.

    Batch axes of `mat` and `vec` are broadcast against each other.

    Differentiable with respect to both `mat` and `vec`.

    Parameters
    ----------
    mat : torch.Tensor
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`.
    vec : torch.Tensor
        Vectors, shape `(*B, C)`.

    Returns
    -------
    torch.Tensor
        `mat @ vec`, shape `(*B, C)` with the broadcast batch shape.
    """
    check_dtype(mat, vec)
    _check_sym(mat, vec)
    batch = torch.broadcast_shapes(mat.shape[:-1], vec.shape[:-1])
    mat = mat.expand(*batch, mat.shape[-1])
    vec = vec.expand(*batch, vec.shape[-1])
    return _MatVec.apply(mat, vec)


def _matvec_acc(out0: Tensor, mat: Tensor, vec: Tensor, sub: bool) -> Tensor:
    """Shared body of [`sym_addmatvec_`][fastfields.torch.sym_addmatvec_] /
    [`sym_submatvec_`][fastfields.torch.sym_submatvec_].
    """
    name = "sym_submatvec_" if sub else "sym_addmatvec_"
    check_dtype(out0, mat, vec)
    _check_sym(mat, vec)
    if out0.shape[-1] != vec.shape[-1]:
        raise ValueError(
            f"{name}: out0 has {out0.shape[-1]} channels but vec has "
            f"{vec.shape[-1]}"
        )
    check_inplace_leaf(name, out0)
    # out0 fixes the batch shape (it is the caller's buffer); mat and vec are
    # broadcast to it with 0-stride views the stride-aware binding reads
    # directly.
    batch = out0.shape[:-1]
    mat = mat.expand(*batch, mat.shape[-1])
    vec = vec.expand(*batch, vec.shape[-1])
    return _MatVecAcc.apply(out0, mat, vec, sub)


def sym_addmatvec_(out0: Tensor, mat: Tensor, vec: Tensor) -> Tensor:
    """Accumulate a matrix-vector product in place: `out0 += mat @ vec`.

    `out0` fixes the batch shape; `mat` and `vec` are broadcast to it. Note
    that the accumulator comes first, unlike in
    [`sym_matvec`][fastfields.torch.sym_matvec].

    Differentiable with respect to `out0` only. A backward pass that needs the
    gradient with respect to `mat` or `vec` raises `RuntimeError`; use `out0 +
    sym_matvec(mat, vec)` when you need those.

    Parameters
    ----------
    out0 : torch.Tensor
        Accumulator, shape `(*B, C)`. Updated in place. It fixes the batch
        shape of the call.
    mat : torch.Tensor
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`, broadcast to
        the batch shape of `out0`.
    vec : torch.Tensor
        Vectors, shape `(*B, C)`, broadcast to the batch shape of `out0`.

    Returns
    -------
    torch.Tensor
        `out0` itself, now holding `out0 + mat @ vec`.
    """
    return _matvec_acc(out0, mat, vec, sub=False)


def sym_submatvec_(out0: Tensor, mat: Tensor, vec: Tensor) -> Tensor:
    """Subtract a matrix-vector product in place: `out0 -= mat @ vec`.

    The subtracting twin of
    [`sym_addmatvec_`][fastfields.torch.sym_addmatvec_], with the same
    contract.

    Differentiable with respect to `out0` only. A backward pass that needs the
    gradient with respect to `mat` or `vec` raises `RuntimeError`; use `out0 +
    sym_matvec(mat, vec)` when you need those.

    Parameters
    ----------
    out0 : torch.Tensor
        Accumulator, shape `(*B, C)`. Updated in place. It fixes the batch
        shape of the call.
    mat : torch.Tensor
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`, broadcast to
        the batch shape of `out0`.
    vec : torch.Tensor
        Vectors, shape `(*B, C)`, broadcast to the batch shape of `out0`.

    Returns
    -------
    torch.Tensor
        `out0` itself, now holding `out0 - mat @ vec`.
    """
    return _matvec_acc(out0, mat, vec, sub=True)


def sym_solve(
    mat: Tensor, vec: Tensor, weight: Optional[Tensor] = None
) -> Tensor:
    """Solve compact-symmetric systems, optionally with a diagonal added.

    A compact-symmetric matrix with `C` channels is stored along the last axis
    as `C*(C+1)//2` values: the diagonal first, then the rows of the upper
    triangle. Vectors have a last axis of length `C`. Solves `(mat +
    diag(weight)) x = vec` for `x`, voxel by voxel. Batch axes of all inputs
    are broadcast against each other.

    [`sym_solve_`][fastfields.torch.sym_solve_] is the in-place version.

    Differentiable with respect to `vec` only; `weight` is treated as a
    constant. The solve does not backpropagate through `mat`, which therefore
    must not require grad: pass `mat.detach()`.

    Parameters
    ----------
    mat : torch.Tensor
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`.
    vec : torch.Tensor
        Right-hand sides, shape `(*B, C)`.
    weight : torch.Tensor, optional
        Diagonal added to `mat` before solving, shape `(*B, C)`.

    Returns
    -------
    torch.Tensor
        The solution `x`, shape `(*B, C)` with the broadcast batch shape.

    Raises
    ------
    TypeError
        If an argument is not float32 or float64.
    ValueError
        If `mat` requires grad.
    """
    check_dtype(mat, vec)
    if weight is not None:
        check_dtype(weight)
    c = _check_sym(mat, vec)
    if weight is not None and weight.shape[-1] != c:
        raise ValueError(
            f"weight has {weight.shape[-1]} channels but the packed matrix "
            f"encodes {c} channels"
        )
    shapes = [mat.shape[:-1], vec.shape[:-1]]
    if weight is not None:
        shapes.append(weight.shape[:-1])
    batch = torch.broadcast_shapes(*shapes)
    mat = mat.expand(*batch, mat.shape[-1])
    vec = vec.expand(*batch, vec.shape[-1])
    if weight is not None:
        weight = weight.expand(*batch, weight.shape[-1])
    return _Solve.apply(mat, vec, weight)


def sym_solve_(
    inp_out: Tensor, mat: Tensor, weight: Optional[Tensor] = None
) -> Tensor:
    """Solve compact-symmetric linear systems in place.

    Solves `(mat + diag(weight)) x = inp_out` for `x` and stores `x` in
    `inp_out`. `inp_out` fixes the batch shape; `mat` and `weight` are
    broadcast to it. Note that the right-hand side comes first, unlike in
    [`sym_solve`][fastfields.torch.sym_solve].

    Differentiable with respect to `inp_out`, like
    [`sym_solve`][fastfields.torch.sym_solve], whose backward pass never reads
    the overwritten right-hand side. `weight` is a constant and `mat` must not
    require grad. A leaf tensor that requires grad cannot be modified in place.

    Parameters
    ----------
    inp_out : torch.Tensor
        Right-hand sides, shape `(*B, C)`, on entry; the solutions on exit.
    mat : torch.Tensor
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`, broadcast to
        the batch shape of `inp_out`.
    weight : torch.Tensor, optional
        Diagonal added to `mat`, shape `(*B, C)`, broadcast to the batch
        shape of `inp_out`.

    Returns
    -------
    torch.Tensor
        `inp_out` itself, now holding the solution.

    Raises
    ------
    ValueError
        If `mat` requires grad, or the channel counts disagree.
    RuntimeError
        If `inp_out` is a leaf tensor that requires grad.
    """
    check_dtype(mat, inp_out)
    check_inplace_leaf("sym_solve_", inp_out)
    if weight is not None:
        check_dtype(weight)
    c = _check_sym(mat, inp_out)
    if weight is not None and weight.shape[-1] != c:
        raise ValueError(
            f"weight has {weight.shape[-1]} channels but the packed matrix "
            f"encodes {c} channels"
        )
    batch = inp_out.shape[:-1]
    mat = mat.expand(*batch, mat.shape[-1])
    if weight is not None:
        weight = weight.expand(*batch, weight.shape[-1])
    return _SolveInPlace.apply(inp_out, mat, weight)


def sym_invert(mat: Tensor) -> Tensor:
    """Invert compact-symmetric matrices.

    A compact-symmetric matrix with `C` channels is stored along the last axis
    as `C*(C+1)//2` values: the diagonal first, then the rows of the upper
    triangle. Vectors have a last axis of length `C`. The inverse is symmetric
    too and is returned in the same packing.

    [`sym_invert_`][fastfields.torch.sym_invert_] is the in-place version.

    Not differentiable: the forward pass runs normally, even when an input
    requires grad, but calling `.backward()` through the result raises
    `RuntimeError`.

    Parameters
    ----------
    mat : torch.Tensor
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`.

    Returns
    -------
    torch.Tensor
        The packed inverses, with the shape of `mat`.

    Raises
    ------
    RuntimeError
        If `.backward()` is called through the result.
    """
    check_dtype(mat)
    return _Invert.apply(mat)


def sym_invert_(mat: Tensor) -> Tensor:
    """Invert compact-symmetric matrices in place.

    The inverse is symmetric too and replaces `mat` in the same packing.

    Not differentiable: the forward pass runs normally, even when an input
    requires grad, but calling `.backward()` through the result raises
    `RuntimeError`. A leaf tensor that requires grad cannot be modified in
    place; this is checked before `mat` is touched.

    Parameters
    ----------
    mat : torch.Tensor
        Compact-symmetric matrices, shape `(*B, C*(C+1)//2)`, on entry;
        their inverses on exit.

    Returns
    -------
    torch.Tensor
        `mat` itself, now holding the packed inverses.

    Raises
    ------
    RuntimeError
        If `.backward()` is called through the result, or `mat` is a leaf
        tensor that requires grad.
    """
    check_dtype(mat)
    check_inplace_leaf("sym_invert_", mat)
    return _InvertInPlace.apply(mat)


class _MatVec(torch.autograd.Function):
    """Autograd for ``out = mat @ vec`` (mirrors jitfields ``MatVec``)."""

    @staticmethod
    def forward(ctx, mat, vec):
        # mat/vec may be 0-stride broadcast views (from Tensor.expand); the
        # stride-aware binding consumes them zero-copy, so we do NOT force
        # contiguity. Only the output must be a real contiguous buffer.
        out = vec.new_empty(vec.shape)
        _fb.sym_matvec(out, mat, vec, stream=stream_ptr(out))
        ctx.save_for_backward(mat, vec)
        return out

    @staticmethod
    def backward(ctx, grad):
        mat, vec = ctx.saved_tensors
        gmat = gvec = None
        # grad_vec = mat @ grad  (mat is symmetric, hence self-adjoint)
        if ctx.needs_input_grad[1]:
            gvec = grad.new_empty(vec.shape)
            _fb.sym_matvec(gvec, mat, grad, stream=stream_ptr(gvec))
        # grad_mat = outer-product contribution, via the dedicated backward op.
        # gmat carries mat's (possibly expanded) shape; autograd's expand
        # backward then sums it down to the caller's original mat shape.
        if ctx.needs_input_grad[0]:
            gmat = grad.new_empty(mat.shape)
            _fb.sym_matvec_backward(gmat, grad, vec, stream=stream_ptr(gmat))
        return gmat, gvec


class _MatVecAcc(torch.autograd.Function):
    """``out0 (+/-)= mat @ vec`` in place; differentiable wrt ``out0`` only.

    The gradient wrt the accumulator is the identity pass-through, so the
    useful case -- accumulating into a tensor that is itself part of a graph
    -- works. Gradients wrt ``mat``/``vec`` are *not* implemented here even
    though `_MatVec` has them: an in-place accumulate consumes
    ``out0``'s storage, and plumbing the matvec adjoints through a
    ``mark_dirty`` node correctly (so that ``out0``'s own history and the
    matvec's contribution are both honoured) is a separate piece of work
    from exposing the op. Until it is done, calling ``.backward()`` with
    ``mat`` or ``vec`` requiring grad raises rather than silently returning a
    wrong (zero) gradient -- which is why this does not simply return
    ``None`` for them.

    Not a mathematical obstruction: for ``y = out0 +/- M v`` the gradients are
    ``dL/dout0 = g``, ``dL/dv = +/- M g`` (M self-adjoint) and ``dL/dM`` the
    usual outer product, i.e. exactly what `_MatVec` already computes.
    The out-of-place spelling is ``out0 +/- sym_matvec(mat, vec)``, which is
    fully differentiable today.
    """

    @staticmethod
    def forward(ctx, out0, mat, vec, sub):
        fn = _fb.sym_submatvec_ if sub else _fb.sym_addmatvec_
        fn(out0, mat, vec, stream=stream_ptr(out0))
        ctx.needs_matvec_grad = (
            ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        )
        ctx.mark_dirty(out0)
        return out0

    @staticmethod
    def backward(ctx, grad):
        if ctx.needs_matvec_grad:
            raise_not_differentiable(
                "sym_addmatvec_/sym_submatvec_",
                "the in-place accumulate has no gradient wrt `mat`/`vec` on "
                "this backend; use `out0 + sym_matvec(mat, vec)`, which is "
                "differentiable wrt all three.",
            )
        gout = grad if ctx.needs_input_grad[0] else None
        return gout, None, None, None


class _Solve(torch.autograd.Function):
    """Autograd for ``out = (mat + diag(weight)) \\ vec``.

    Mirrors jitfields ``Solve``: differentiable through ``vec`` only. Since the
    system matrix ``M`` is symmetric, ``M^{-1}`` is self-adjoint, so the
    gradient w.r.t. ``vec`` is another solve with the same matrix/weight.
    """

    @staticmethod
    def forward(ctx, mat, vec, weight):
        if mat.requires_grad:
            raise ValueError(
                "sym_solve does not backpropagate gradients through the "
                "matrix. Use `mat.detach()`."
            )
        # mat/vec/weight may be 0-stride broadcast views (Tensor.expand); the
        # stride-aware binding handles them zero-copy. Output is contiguous.
        out = vec.new_empty(vec.shape)
        s = stream_ptr(out)
        if weight is None:
            _fb.sym_solve(out, mat, vec, stream=s)
        else:
            _fb.sym_solve(out, mat, vec, weight, stream=s)
        ctx.save_for_backward(mat, weight)
        return out

    @staticmethod
    def backward(ctx, grad):
        mat, weight = ctx.saved_tensors
        gvec = None
        if ctx.needs_input_grad[1]:
            gvec = grad.new_empty(grad.shape)
            s = stream_ptr(gvec)
            if weight is None:
                _fb.sym_solve(gvec, mat, grad, stream=s)
            else:
                _fb.sym_solve(gvec, mat, grad, weight, stream=s)
        return None, gvec, None


class _SolveInPlace(torch.autograd.Function):
    """``inp_out <- (mat + diag(weight)) \\ inp_out`` in place.

    Autograd-safe: the backward below is identical to `_Solve`'s and
    never reads the pre-mutation ``inp_out`` -- only the saved ``mat``/
    ``weight`` -- so mutating ``inp_out`` in place loses nothing backward
    needs. ``ctx.mark_dirty(inp_out)`` bumps its version counter so that if
    some other op had saved ``inp_out`` for its own backward, autograd
    raises instead of silently computing a wrong gradient.
    """

    @staticmethod
    def forward(ctx, inp_out, mat, weight):
        if mat.requires_grad:
            raise ValueError(
                "sym_solve_ does not backpropagate gradients through the "
                "matrix. Use `mat.detach()`."
            )
        s = stream_ptr(inp_out)
        _fb.sym_solve_(inp_out, mat, weight, stream=s)
        ctx.mark_dirty(inp_out)
        ctx.save_for_backward(mat, weight)
        return inp_out

    @staticmethod
    def backward(ctx, grad):
        mat, weight = ctx.saved_tensors
        ginp = None
        if ctx.needs_input_grad[0]:
            ginp = grad.new_empty(grad.shape)
            s = stream_ptr(ginp)
            if weight is None:
                _fb.sym_solve(ginp, mat, grad, stream=s)
            else:
                _fb.sym_solve(ginp, mat, grad, weight, stream=s)
        return ginp, None, None


class _Invert(torch.autograd.Function):
    """``out = inv(mat)`` (out-of-place, ``mat`` left untouched)."""

    @staticmethod
    def forward(ctx, mat):
        # mat is a read-only input: the stride-aware binding reads it
        # zero-copy, so we do NOT force contiguity. The output must be a
        # real contiguous buffer (new_empty is contiguous even when mat is
        # strided).
        out = mat.new_empty(mat.shape)
        _fb.sym_invert(out, mat, stream=stream_ptr(mat))
        return out

    @staticmethod
    def backward(ctx, grad):
        raise_not_differentiable(
            "sym_invert",
            "no gradient is implemented for the packed-matrix inverse on "
            "this backend (mirrors jitfields). This is an implementation "
            "gap, not a mathematical one.",
        )


class _InvertInPlace(torch.autograd.Function):
    """``mat <- inv(mat)`` in place."""

    @staticmethod
    def forward(ctx, mat):
        _fb.sym_invert_(mat, stream=stream_ptr(mat))
        ctx.mark_dirty(mat)
        return mat

    @staticmethod
    def backward(ctx, grad):
        raise_not_differentiable(
            "sym_invert_",
            "no gradient is implemented for the packed-matrix inverse on "
            "this backend (mirrors jitfields). This is an implementation "
            "gap, not a mathematical one.",
        )
