"""Spatial regularisers for PyTorch.

The same `field_*` and `flow_*` families as `fastfields.numpy`, with matching
signatures so that `fastfields.auto` can dispatch to either.

The `*_matvec` operators are symmetric, so their backward pass applies the
same operator. The diagonal and stencil builders take no tensor input and are
not differentiable. `*_forward` and `*_precond` are compositions of other
differentiable operations rather than calls to the fused kernels of numpy and
cupy; the reason is given in their docstrings.
"""

from __future__ import annotations

from typing import Optional, Sequence

import fastfields.dlpack as _fb

import torch
from fastfields.helpers import as_bound
from torch import Tensor

from ._sym import sym_matvec, sym_solve
from ._util import check_dtype, stream_ptr

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


class _FieldMatvec(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, inp, voxel_size, absolute, membrane, bending, bound, ndim
    ):
        out = inp.new_zeros(inp.shape)
        _fb.field_matvec(
            out,
            inp,
            voxel_size=voxel_size,
            absolute=absolute,
            membrane=membrane,
            bending=bending,
            bound=bound,
            ndim=ndim,
            stream=stream_ptr(out),
        )
        ctx.args = (voxel_size, absolute, membrane, bending, bound, ndim)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ginp = None
        if ctx.needs_input_grad[0]:
            vs, ab, mem, ben, bnd, ndim = ctx.args
            ginp = grad_out.new_zeros(grad_out.shape)
            _fb.field_matvec(
                ginp,
                grad_out.contiguous(),
                voxel_size=vs,
                absolute=ab,
                membrane=mem,
                bending=ben,
                bound=bnd,
                ndim=ndim,
                stream=stream_ptr(ginp),
            )
        return ginp, None, None, None, None, None, None


class _FlowMatvec(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp,
        voxel_size,
        absolute,
        membrane,
        bending,
        shears,
        div,
        bound,
        ndim,
    ):
        out = inp.new_zeros(inp.shape)
        _fb.flow_matvec(
            out,
            inp,
            voxel_size=voxel_size,
            absolute=absolute,
            membrane=membrane,
            bending=bending,
            shears=shears,
            div=div,
            bound=bound,
            ndim=ndim,
            stream=stream_ptr(out),
        )
        ctx.args = (
            voxel_size,
            absolute,
            membrane,
            bending,
            shears,
            div,
            bound,
            ndim,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # The flow regulariser operator is self-adjoint, so the adjoint of the
        # forward matvec is the same matvec applied to grad_out.
        ginp = None
        if ctx.needs_input_grad[0]:
            vs, ab, mem, ben, sh, dv, bnd, ndim = ctx.args
            ginp = grad_out.new_zeros(grad_out.shape)
            _fb.flow_matvec(
                ginp,
                grad_out.contiguous(),
                voxel_size=vs,
                absolute=ab,
                membrane=mem,
                bending=ben,
                shears=sh,
                div=dv,
                bound=bnd,
                ndim=ndim,
                stream=stream_ptr(ginp),
            )
        return ginp, None, None, None, None, None, None, None, None


class _FieldMatvecRLS(torch.autograd.Function):
    """RLS/JRLS-weighted variant of ``_FieldMatvec``.

    Differentiable wrt ``inp`` only -- ``wgt`` is treated as a fixed
    coefficient field (like ``bound``/``ndim``), exactly as ``field_diag``
    has no differentiable input. For a *fixed* weight map, ``L(w)`` is still
    self-adjoint, so the backward pass is the same weighted matvec applied to
    ``grad_out``.
    """

    @staticmethod
    def forward(
        ctx, inp, wgt, voxel_size, absolute, membrane, bending, bound, ndim
    ):
        out = inp.new_zeros(inp.shape)
        _fb.field_matvec_rls(
            out,
            inp,
            wgt,
            voxel_size=voxel_size,
            absolute=absolute,
            membrane=membrane,
            bending=bending,
            bound=bound,
            ndim=ndim,
            stream=stream_ptr(out),
        )
        ctx.save_for_backward(wgt)
        ctx.args = (voxel_size, absolute, membrane, bending, bound, ndim)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ginp = None
        if ctx.needs_input_grad[0]:
            (wgt,) = ctx.saved_tensors
            vs, ab, mem, ben, bnd, ndim = ctx.args
            ginp = grad_out.new_zeros(grad_out.shape)
            _fb.field_matvec_rls(
                ginp,
                grad_out.contiguous(),
                wgt,
                voxel_size=vs,
                absolute=ab,
                membrane=mem,
                bending=ben,
                bound=bnd,
                ndim=ndim,
                stream=stream_ptr(ginp),
            )
        return ginp, None, None, None, None, None, None, None


class _FlowMatvecRLS(torch.autograd.Function):
    """JRLS-weighted variant of ``_FlowMatvec``.

    Differentiable wrt ``inp`` only -- ``wgt`` is treated as a fixed
    coefficient field, exactly as in ``_FieldMatvecRLS``. For a *fixed*
    weight map ``L(w)`` is still self-adjoint, so the backward pass is the
    same weighted matvec applied to ``grad_out``.
    """

    @staticmethod
    def forward(
        ctx,
        inp,
        wgt,
        voxel_size,
        absolute,
        membrane,
        bending,
        shears,
        div,
        bound,
        ndim,
    ):
        out = inp.new_zeros(inp.shape)
        _fb.flow_matvec_rls(
            out,
            inp,
            wgt,
            voxel_size=voxel_size,
            absolute=absolute,
            membrane=membrane,
            bending=bending,
            shears=shears,
            div=div,
            bound=bound,
            ndim=ndim,
            stream=stream_ptr(out),
        )
        ctx.save_for_backward(wgt)
        ctx.args = (
            voxel_size,
            absolute,
            membrane,
            bending,
            shears,
            div,
            bound,
            ndim,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ginp = None
        if ctx.needs_input_grad[0]:
            (wgt,) = ctx.saved_tensors
            vs, ab, mem, ben, sh, dv, bnd, ndim = ctx.args
            ginp = grad_out.new_zeros(grad_out.shape)
            _fb.flow_matvec_rls(
                ginp,
                grad_out.contiguous(),
                wgt,
                voxel_size=vs,
                absolute=ab,
                membrane=mem,
                bending=ben,
                shears=sh,
                div=dv,
                bound=bnd,
                ndim=ndim,
                stream=stream_ptr(ginp),
            )
        return ginp, None, None, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# In-place accumulate ops (`out (+/-)= ...`)
# ---------------------------------------------------------------------------
#
# These wrap the *in-place-only* C primitives `ff::{field,flow}_{matvec,diag}_
# {add,sub}_`, which read-modify-write the caller's buffer. The out-of-place
# spelling here is a `.clone()` followed by that very same primitive --
# there is only one kernel, exactly as in jitfields.
#
# Autograd safety (see API_CONTRACT.md "In-place policy"):
#
#   The operation is `acc <- acc (+/-) g(...)`, i.e. *additive* in the
#   tensor it mutates. Therefore
#       d(acc +/- g)/d(acc) = I          (identity -- independent of acc)
#       d(acc +/- g)/d(inp) = +/- L      (L linear & self-adjoint)
#   Neither partial derivative depends on the *pre-mutation* value of `acc`, so
#   nothing has to be saved for backward and overwriting `acc` destroys no
#   information the backward pass needs. This is precisely why PyTorch itself
#   ships `Tensor.add_` as a fully autograd-safe in-place op.
#
#   Note what is (deliberately) absent below: no `ctx.save_for_backward`. That
#   absence *is* the safety argument, structurally. What IS required is
#   `ctx.mark_dirty(acc)`, which bumps `acc`'s version counter so that if any
#   other op had saved `acc` for *its* backward, autograd raises instead of
#   silently returning a wrong gradient.
#
#   Torch's ordinary leaf rule still applies: a leaf tensor with
#   `requires_grad=True` cannot be mutated in place (identically to
#   `x.add_(y)`); use the out-of-place spelling for those.


class _FieldMatvecAcc(torch.autograd.Function):
    """``acc (+/-)= L @ inp`` in place; diff'able wrt ``acc``/``inp``."""

    @staticmethod
    def forward(
        ctx,
        acc,
        inp,
        voxel_size,
        absolute,
        membrane,
        bending,
        bound,
        ndim,
        sub,
    ):
        fn = _fb.field_submatvec_ if sub else _fb.field_addmatvec_
        fn(
            acc,
            inp,
            voxel_size=voxel_size,
            absolute=absolute,
            membrane=membrane,
            bending=bending,
            bound=bound,
            ndim=ndim,
            stream=stream_ptr(acc),
        )
        # Required for in-place autograd bookkeeping (version counter).
        ctx.mark_dirty(acc)
        ctx.args = (voxel_size, absolute, membrane, bending, bound, ndim)
        ctx.sub = sub
        return acc

    @staticmethod
    def backward(ctx, grad_out):
        # d/d(acc) is the identity, so the incoming gradient passes straight
        # through -- no pre-mutation value of `acc` is involved.
        gacc = grad_out if ctx.needs_input_grad[0] else None

        ginp = None
        if ctx.needs_input_grad[1]:
            vs, ab, mem, ben, bnd, ndim = ctx.args
            ginp = grad_out.new_zeros(grad_out.shape)
            # L is self-adjoint, so the adjoint of `inp -> L @ inp` is itself.
            _fb.field_matvec(
                ginp,
                grad_out.contiguous(),
                voxel_size=vs,
                absolute=ab,
                membrane=mem,
                bending=ben,
                bound=bnd,
                ndim=ndim,
                stream=stream_ptr(ginp),
            )
            if ctx.sub:
                ginp = ginp.neg()

        return gacc, ginp, None, None, None, None, None, None, None


class _FlowMatvecAcc(torch.autograd.Function):
    """``acc (+/-)= L @ inp`` in place, flow; diff'able wrt acc/inp."""

    @staticmethod
    def forward(
        ctx,
        acc,
        inp,
        voxel_size,
        absolute,
        membrane,
        bending,
        shears,
        div,
        bound,
        ndim,
        sub,
    ):
        fn = _fb.flow_submatvec_ if sub else _fb.flow_addmatvec_
        fn(
            acc,
            inp,
            voxel_size=voxel_size,
            absolute=absolute,
            membrane=membrane,
            bending=bending,
            shears=shears,
            div=div,
            bound=bound,
            ndim=ndim,
            stream=stream_ptr(acc),
        )
        ctx.mark_dirty(acc)
        ctx.args = (
            voxel_size,
            absolute,
            membrane,
            bending,
            shears,
            div,
            bound,
            ndim,
        )
        ctx.sub = sub
        return acc

    @staticmethod
    def backward(ctx, grad_out):
        gacc = grad_out if ctx.needs_input_grad[0] else None

        ginp = None
        if ctx.needs_input_grad[1]:
            vs, ab, mem, ben, sh, dv, bnd, ndim = ctx.args
            ginp = grad_out.new_zeros(grad_out.shape)
            _fb.flow_matvec(
                ginp,
                grad_out.contiguous(),
                voxel_size=vs,
                absolute=ab,
                membrane=mem,
                bending=ben,
                shears=sh,
                div=dv,
                bound=bnd,
                ndim=ndim,
                stream=stream_ptr(ginp),
            )
            if ctx.sub:
                ginp = ginp.neg()

        return (
            gacc,
            ginp,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _FieldDiagAcc(torch.autograd.Function):
    """``acc (+/-)= diag(L)`` in place; differentiable wrt ``acc``.

    ``diag(L)`` is a function of the shape and the penalty weights only -- it
    has no tensor input at all -- so the only gradient is the identity
    pass-through wrt ``acc``.
    """

    @staticmethod
    def forward(
        ctx, acc, voxel_size, absolute, membrane, bending, bound, ndim, sub
    ):
        fn = _fb.field_subdiag_ if sub else _fb.field_adddiag_
        fn(
            acc,
            voxel_size=voxel_size,
            absolute=absolute,
            membrane=membrane,
            bending=bending,
            bound=bound,
            ndim=ndim,
            stream=stream_ptr(acc),
        )
        ctx.mark_dirty(acc)
        return acc

    @staticmethod
    def backward(ctx, grad_out):
        gacc = grad_out if ctx.needs_input_grad[0] else None
        return gacc, None, None, None, None, None, None, None


class _FieldKernelAcc(torch.autograd.Function):
    """``acc (+/-)= K`` in place; differentiable wrt ``acc``.

    ``K`` is the materialised Toeplitz stencil of the field regulariser: like
    ``diag(L)`` it is a function of the penalty weights and the voxel size
    only, with no tensor input, so the only gradient is the identity
    pass-through wrt ``acc``. Same shape as
    [`field_kernel`][fastfields.torch.field_kernel]'s output.
    """

    @staticmethod
    def forward(
        ctx, acc, voxel_size, absolute, membrane, bending, bound, ndim, sub
    ):
        fn = _fb.field_subkernel_ if sub else _fb.field_addkernel_
        fn(
            acc,
            voxel_size=voxel_size,
            absolute=absolute,
            membrane=membrane,
            bending=bending,
            bound=bound,
            ndim=ndim,
            stream=stream_ptr(acc),
        )
        ctx.mark_dirty(acc)
        return acc

    @staticmethod
    def backward(ctx, grad_out):
        gacc = grad_out if ctx.needs_input_grad[0] else None
        return gacc, None, None, None, None, None, None, None


class _FlowKernelAcc(torch.autograd.Function):
    """``acc (+/-)= K`` in place (flow); differentiable wrt ``acc``."""

    @staticmethod
    def forward(
        ctx,
        acc,
        voxel_size,
        absolute,
        membrane,
        bending,
        shears,
        div,
        bound,
        ndim,
        sub,
    ):
        fn = _fb.flow_subkernel_ if sub else _fb.flow_addkernel_
        fn(
            acc,
            voxel_size=voxel_size,
            absolute=absolute,
            membrane=membrane,
            bending=bending,
            shears=shears,
            div=div,
            bound=bound,
            ndim=ndim,
            stream=stream_ptr(acc),
        )
        ctx.mark_dirty(acc)
        return acc

    @staticmethod
    def backward(ctx, grad_out):
        gacc = grad_out if ctx.needs_input_grad[0] else None
        return (gacc,) + (None,) * 9


class _FlowDiagAcc(torch.autograd.Function):
    """``acc (+/-)= diag(L)`` in place (flow); differentiable wrt ``acc``."""

    @staticmethod
    def forward(
        ctx,
        acc,
        voxel_size,
        absolute,
        membrane,
        bending,
        shears,
        div,
        bound,
        ndim,
        sub,
    ):
        fn = _fb.flow_subdiag_ if sub else _fb.flow_adddiag_
        fn(
            acc,
            voxel_size=voxel_size,
            absolute=absolute,
            membrane=membrane,
            bending=bending,
            shears=shears,
            div=div,
            bound=bound,
            ndim=ndim,
            stream=stream_ptr(acc),
        )
        ctx.mark_dirty(acc)
        return acc

    @staticmethod
    def backward(ctx, grad_out):
        gacc = grad_out if ctx.needs_input_grad[0] else None
        return gacc, None, None, None, None, None, None, None, None, None


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
    """Single entry point behind the four field-matvec accumulate spellings."""
    check_dtype(inp)
    check_dtype(field)
    channels = field.shape[-1]
    acc = inp if inplace else inp.clone()
    return _FieldMatvecAcc.apply(
        acc,
        field,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        sub,
    )


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
    """Single entry point behind the four flow-matvec accumulate spellings."""
    check_dtype(inp)
    check_dtype(flow)
    acc = inp if inplace else inp.clone()
    return _FlowMatvecAcc.apply(
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
        sub,
    )


def _field_diag_acc(
    inp, absolute, membrane, bending, voxel_size, bound, ndim, sub, inplace
):
    """Single entry point behind the four field-diag accumulate spellings."""
    check_dtype(inp)
    channels = inp.shape[-1]
    acc = inp if inplace else inp.clone()
    return _FieldDiagAcc.apply(
        acc,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        sub,
    )


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
    """Single entry point behind the four flow-diag accumulate spellings."""
    check_dtype(inp)
    acc = inp if inplace else inp.clone()
    return _FlowDiagAcc.apply(
        acc,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
        sub,
    )


def _field_kernel_acc(
    inp, absolute, membrane, bending, voxel_size, bound, ndim, sub, inplace
):
    """Single entry point behind the four field-kernel accumulate spellings."""
    check_dtype(inp)
    channels = inp.shape[-1]
    acc = inp if inplace else inp.clone()
    return _FieldKernelAcc.apply(
        acc,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        sub,
    )


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
    """Single entry point behind the four flow-kernel accumulate spellings."""
    check_dtype(inp)
    acc = inp if inplace else inp.clone()
    return _FlowKernelAcc.apply(
        acc,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
        sub,
    )


def field_matvec(
    inp: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Apply the field regulariser to a multi-channel field.

    Computes `L @ inp` for the field regulariser `L`, the sum of the requested
    penalties. Each penalty is given per channel and the regulariser never
    couples channels. The highest-order penalty that is set decides the
    finite-difference stencil.

    With unit voxels and only `absolute`, the operator scales each channel by
    its weight; with only `membrane` it is the discrete negative Laplacian.

    The `*_addmatvec`, `*_submatvec` and `*_rls` families accumulate or
    reweight this operator.

    Differentiable with respect to `inp`. The operator is symmetric, so the
    backward pass applies the same operator.

    Parameters
    ----------
    inp : torch.Tensor
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
    torch.Tensor
        `L @ inp`, a new tensor the shape of `inp`.
    """
    check_dtype(inp)
    channels = inp.shape[-1]
    return _FieldMatvec.apply(
        inp,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
    )


def flow_matvec(
    inp: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Apply the flow regulariser to a vector flow field.

    Computes `L @ inp` for the flow regulariser `L`, the sum of the requested
    penalties. The penalties are scalars. A non-zero `shears` or `div` adds the
    linear-elastic terms, which couple the flow components. The highest-order
    penalty that is set decides the finite-difference stencil.

    With unit voxels and only `absolute`, the operator scales each channel by
    its weight; with only `membrane` it is the discrete negative Laplacian.

    The `*_addmatvec`, `*_submatvec` and `*_rls` families accumulate or
    reweight this operator.

    Differentiable with respect to `inp`. The operator is symmetric, so the
    backward pass applies the same operator.

    Parameters
    ----------
    inp : torch.Tensor
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
    torch.Tensor
        `L @ inp`, a new tensor the shape of `inp`.
    """
    check_dtype(inp)
    return _FlowMatvec.apply(
        inp,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
    )


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
    dtype: torch.dtype = torch.float64,
    device=None,
) -> Tensor:
    """Convolution stencil of the field regulariser.

    Returns the small centred kernel that, convolved with a field, reproduces
    [`field_matvec`][fastfields.torch.field_matvec] with the same penalties.
    Its shape is `(*k, C)`: one independent stencil per channel. `k` is the
    stencil width along each spatial axis: 1 if only `absolute` is set, 3 with
    `membrane`, 5 with `bending`.

    Not differentiable: it builds a result from a shape and takes no tensor
    input.

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
    dtype : dtype, default=torch.float64
        Data type of the result.
    device : torch.device, optional
        Device of the result; torch's default device if omitted.

    Returns
    -------
    torch.Tensor
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
    out = torch.zeros(
        tuple([width] * ndim + [channels]), dtype=dtype, device=device
    )
    _fb.field_kernel(
        out,
        voxel_size=_voxel(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=as_bound(bound),
        ndim=ndim,
        stream=stream_ptr(out),
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
    dtype: torch.dtype = torch.float64,
    device=None,
) -> Tensor:
    """Diagonal of the field regulariser.

    Builds the diagonal of the field regulariser `L`, which is a cheap Jacobi
    preconditioner. Each penalty is given per channel and the regulariser never
    couples channels.

    The result is allocated from `shape`; for a version that accumulates onto
    an existing array see [`field_adddiag`][fastfields.torch.field_adddiag].

    Not differentiable: it builds a result from a shape and takes no tensor
    input.

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
    dtype : dtype, default=torch.float64
        Data type of the result.
    device : torch.device, optional
        Device of the result; torch's default device if omitted.

    Returns
    -------
    torch.Tensor
        The diagonal of `L`, shape `shape`.
    """
    out = torch.zeros(tuple(int(s) for s in shape), dtype=dtype, device=device)
    channels = out.shape[-1]
    _fb.field_diag(
        out,
        voxel_size=_voxel(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=as_bound(bound),
        ndim=ndim,
        stream=stream_ptr(out),
    )
    return out


def field_relax(
    field: Tensor,
    hes: Tensor,
    grd: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    nb_iter: int = 1,
) -> Tensor:
    """Solve the regularised system `(H + L) x = g` by relaxation sweeps.

    Runs `nb_iter` Gauss-Seidel sweeps on `(H + L) x = g`, where `H` is the
    per-voxel Hessian `hes` and `L` is the field regulariser `L`. Each penalty
    is given per channel and the regulariser never couples channels.

    `field` is the starting point and receives the result: it is updated in
    place and returned.

    Not differentiable, being an iterative in-place solver.

    Parameters
    ----------
    field : torch.Tensor
        Starting point, shape `(*batch, *spatial, C)`. Updated in place.
    hes : torch.Tensor
        Per-voxel Hessian `H` in compact-symmetric packing, shape `(*batch,
        *spatial, C*(C+1)//2)`.
    grd : torch.Tensor
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
    torch.Tensor
        `field` itself, holding the refined solution.
    """
    check_dtype(field)
    channels = field.shape[-1]
    _fb.field_relax(
        field,
        hes.contiguous(),
        grd.contiguous(),
        voxel_size=_voxel(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=as_bound(bound),
        ndim=ndim,
        nb_iter=int(nb_iter),
        stream=stream_ptr(field),
    )
    return field


def field_matvec_rls(
    inp: Tensor,
    wgt: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Apply the weighted (RLS/JRLS) field regulariser.

    The reweighted form of [`field_matvec`][fastfields.torch.field_matvec]. The
    weight map `wgt` modulates the strength of the penalties voxel by voxel,
    for edge-preserving or robust regularisation. Its last axis decides the
    mode: 1 for one weight shared by all channels (JRLS), `C` for one per
    channel (RLS).

    Differentiable with respect to `inp`; `wgt` is treated as fixed.

    Parameters
    ----------
    inp : torch.Tensor
        The field to filter, shape `(*batch, *spatial, C)`.
    wgt : torch.Tensor
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
    torch.Tensor
        The filtered result, a new tensor the shape of `inp`.
    """
    check_dtype(inp)
    channels = inp.shape[-1]
    return _FieldMatvecRLS.apply(
        inp,
        wgt,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
    )


def field_diag_rls(
    wgt: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    channels: int | None = None,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> Tensor:
    """Diagonal of the weighted (RLS/JRLS) field regulariser.

    The diagonal of the operator of
    [`field_matvec_rls`][fastfields.torch.field_matvec_rls]. Unlike
    [`field_diag`][fastfields.torch.field_diag] it is shaped by the weight map
    and not by a `shape` argument. The weight map `wgt` modulates the strength
    of the penalties voxel by voxel, for edge-preserving or robust
    regularisation. Its last axis decides the mode: 1 for one weight shared by
    all channels (JRLS), `C` for one per channel (RLS).

    Not differentiable.

    Parameters
    ----------
    wgt : torch.Tensor
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
    dtype : dtype, default=torch.float64
        Data type of the result.
    device : torch.device, optional
        Device of the result; the device of `wgt` if omitted.

    Returns
    -------
    torch.Tensor
        The diagonal, shape `(*batch, *spatial, C)`.
    """
    if channels is None:
        channels = (
            wgt.shape[-1]
            if wgt.shape[-1] != 1
            else _field_channels(None, absolute, membrane, bending)
        )
    out = torch.zeros(
        tuple(wgt.shape[:-1]) + (channels,),
        dtype=dtype,
        device=device if device is not None else wgt.device,
    )
    _fb.field_diag_rls(
        out,
        wgt,
        voxel_size=_voxel(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=as_bound(bound),
        ndim=ndim,
        stream=stream_ptr(out),
    )
    return out


def field_relax_rls(
    field: Tensor,
    hes: Tensor,
    grd: Tensor,
    wgt: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    nb_iter: int = 1,
) -> Tensor:
    """Solve the weighted system `(H + L(w)) x = g` by relaxation sweeps.

    The reweighted form of [`field_relax`][fastfields.torch.field_relax]:
    `nb_iter` Gauss-Seidel sweeps on `(H + L(w)) x = g`, where `L(w)` is the
    weighted regulariser of
    [`field_matvec_rls`][fastfields.torch.field_matvec_rls]. The weight map
    `wgt` modulates the strength of the penalties voxel by voxel, for
    edge-preserving or robust regularisation. Its last axis decides the mode: 1
    for one weight shared by all channels (JRLS), `C` for one per channel
    (RLS).

    `field` is the starting point and receives the result: it is updated in
    place and returned.

    Not differentiable, being an iterative in-place solver.

    Parameters
    ----------
    field : torch.Tensor
        Starting point, shape `(*batch, *spatial, C)`. Updated in place.
    hes : torch.Tensor
        Per-voxel Hessian `H` in compact-symmetric packing, shape `(*batch,
        *spatial, C*(C+1)//2)`.
    grd : torch.Tensor
        Gradient `g`, same shape as `field`.
    wgt : torch.Tensor
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
    torch.Tensor
        `field` itself, holding the refined solution.
    """
    check_dtype(field)
    channels = field.shape[-1]
    _fb.field_relax_rls(
        field,
        hes.contiguous(),
        grd.contiguous(),
        wgt.contiguous(),
        voxel_size=_voxel(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=as_bound(bound),
        ndim=ndim,
        nb_iter=int(nb_iter),
        stream=stream_ptr(field),
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
    dtype: torch.dtype = torch.float64,
    device=None,
) -> Tensor:
    """Diagonal of the flow regulariser.

    Builds the diagonal of the flow regulariser `L`, which is a cheap Jacobi
    preconditioner. The penalties are scalars. A non-zero `shears` or `div`
    adds the linear-elastic terms, which couple the flow components.

    The result is allocated from `shape`; for a version that accumulates onto
    an existing array see [`flow_adddiag`][fastfields.torch.flow_adddiag].

    Not differentiable: it builds a result from a shape and takes no tensor
    input.

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
    dtype : dtype, default=torch.float64
        Data type of the result.
    device : torch.device, optional
        Device of the result; torch's default device if omitted.

    Returns
    -------
    torch.Tensor
        The diagonal of `L`, shape `shape`.
    """
    out = torch.zeros(tuple(int(s) for s in shape), dtype=dtype, device=device)
    _fb.flow_diag(
        out,
        voxel_size=_voxel(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=as_bound(bound),
        ndim=ndim,
        stream=stream_ptr(out),
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
    dtype: torch.dtype = torch.float64,
    device=None,
) -> Tensor:
    """Convolution stencil of the flow regulariser.

    Returns the small centred kernel that, convolved with a flow, reproduces
    [`flow_matvec`][fastfields.torch.flow_matvec] with the same penalties. Its
    shape is `(*k, ndim)` for the per-component stencil, or `(*k, ndim, ndim)`
    when `shears` or `div` select the matrix stencil that couples components.
    `k` is the stencil width along each spatial axis: 1 if only `absolute` is
    set, 3 with `membrane` or the elastic terms, 5 with `bending`.

    Not differentiable: it builds a result from a shape and takes no tensor
    input.

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
    dtype : dtype, default=torch.float64
        Data type of the result.
    device : torch.device, optional
        Device of the result; torch's default device if omitted.

    Returns
    -------
    torch.Tensor
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
    out = torch.zeros(tuple(shape), dtype=dtype, device=device)
    _fb.flow_kernel(
        out,
        voxel_size=_voxel(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=as_bound(bound),
        ndim=ndim,
        stream=stream_ptr(out),
    )
    return out


def flow_relax(
    flow: Tensor,
    hes: Tensor,
    grd: Tensor,
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
) -> Tensor:
    """Solve the regularised system `(H + L) x = g` by relaxation sweeps.

    Runs `nb_iter` Gauss-Seidel sweeps on `(H + L) x = g`, where `H` is the
    per-voxel Hessian `hes` and `L` is the flow regulariser `L`. The penalties
    are scalars. A non-zero `shears` or `div` adds the linear-elastic terms,
    which couple the flow components.

    `flow` is the starting point and receives the result: it is updated in
    place and returned.

    Not differentiable, being an iterative in-place solver.

    Parameters
    ----------
    flow : torch.Tensor
        Starting point, shape `(*batch, *spatial, ndim)`. Updated in place.
    hes : torch.Tensor
        Per-voxel Hessian `H` in compact-symmetric packing, shape `(*batch,
        *spatial, ndim*(ndim+1)//2)`.
    grd : torch.Tensor
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
    torch.Tensor
        `flow` itself, holding the refined solution.
    """
    check_dtype(flow)
    _fb.flow_relax(
        flow,
        hes.contiguous(),
        grd.contiguous(),
        voxel_size=_voxel(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=as_bound(bound),
        ndim=ndim,
        nb_iter=int(nb_iter),
        stream=stream_ptr(flow),
    )
    return flow


def flow_matvec_rls(
    inp: Tensor,
    wgt: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Apply the weighted (RLS/JRLS) flow regulariser.

    The reweighted form of [`flow_matvec`][fastfields.torch.flow_matvec]. The
    weight map `wgt` modulates the strength of the penalties voxel by voxel,
    for edge-preserving or robust regularisation. Flow weighting is always
    joint: one weight per voxel, shared by all flow components. `bending` is
    not supported with weighting, and a non-zero value is rejected.

    Differentiable with respect to `inp`; `wgt` is treated as fixed.

    Parameters
    ----------
    inp : torch.Tensor
        The flow to filter, shape `(*batch, *spatial, ndim)`.
    wgt : torch.Tensor
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
    torch.Tensor
        The filtered result, a new tensor the shape of `inp`.
    """
    check_dtype(inp)
    return _FlowMatvecRLS.apply(
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
    )


def flow_diag_rls(
    wgt: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> Tensor:
    """Diagonal of the weighted (JRLS) flow regulariser.

    The diagonal of the operator of
    [`flow_matvec_rls`][fastfields.torch.flow_matvec_rls]. Unlike
    [`flow_diag`][fastfields.torch.flow_diag] it is shaped by the weight map
    and not by a `shape` argument. The weight map `wgt` modulates the strength
    of the penalties voxel by voxel, for edge-preserving or robust
    regularisation. Flow weighting is always joint: one weight per voxel,
    shared by all flow components. `bending` is not supported with weighting,
    and a non-zero value is rejected.

    Not differentiable.

    Parameters
    ----------
    wgt : torch.Tensor
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
    dtype : dtype, default=torch.float64
        Data type of the result.
    device : torch.device, optional
        Device of the result; the device of `wgt` if omitted.

    Returns
    -------
    torch.Tensor
        The diagonal, shape `(*batch, *spatial, ndim)`: the shape of `wgt`
        with its last axis replaced by the number of flow components.
    """
    out = torch.zeros(
        tuple(wgt.shape[:-1]) + (int(ndim),),
        dtype=dtype,
        device=device if device is not None else wgt.device,
    )
    _fb.flow_diag_rls(
        out,
        wgt,
        voxel_size=_voxel(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=as_bound(bound),
        ndim=ndim,
        stream=stream_ptr(out),
    )
    return out


def flow_relax_rls(
    flow: Tensor,
    hes: Tensor,
    grd: Tensor,
    wgt: Tensor,
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
) -> Tensor:
    """Solve the weighted system `(H + L(w)) x = g` by relaxation sweeps.

    The reweighted form of [`flow_relax`][fastfields.torch.flow_relax]:
    `nb_iter` Gauss-Seidel sweeps on `(H + L(w)) x = g`, where `L(w)` is the
    weighted regulariser of
    [`flow_matvec_rls`][fastfields.torch.flow_matvec_rls]. The weight map `wgt`
    modulates the strength of the penalties voxel by voxel, for edge-preserving
    or robust regularisation. Flow weighting is always joint: one weight per
    voxel, shared by all flow components. `bending` is not supported with
    weighting, and a non-zero value is rejected.

    `flow` is the starting point and receives the result: it is updated in
    place and returned.

    Not differentiable, being an iterative in-place solver.

    Parameters
    ----------
    flow : torch.Tensor
        Starting point, shape `(*batch, *spatial, ndim)`. Updated in place.
    hes : torch.Tensor
        Per-voxel Hessian `H` in compact-symmetric packing, shape `(*batch,
        *spatial, ndim*(ndim+1)//2)`.
    grd : torch.Tensor
        Gradient `g`, same shape as `flow`.
    wgt : torch.Tensor
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
    torch.Tensor
        `flow` itself, holding the refined solution.
    """
    check_dtype(flow)
    _fb.flow_relax_rls(
        flow,
        hes.contiguous(),
        grd.contiguous(),
        wgt.contiguous(),
        voxel_size=_voxel(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=as_bound(bound),
        ndim=ndim,
        nb_iter=int(nb_iter),
        stream=stream_ptr(flow),
    )
    return flow


def flow_precond(
    mat: Tensor,
    vec: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Jacobi preconditioner: solve `(M + diag(L)) x = vec` voxel by voxel.

    Solves `(M + diag(L)) x = vec` voxel by voxel, where `M` is the per-voxel
    matrix `mat` and `diag(L)` is the diagonal of the flow regulariser `L`. It
    approximates the inverse of the operator of
    [`flow_forward`][fastfields.torch.flow_forward]. The penalties are scalars.
    A non-zero `shears` or `div` adds the linear-elastic terms, which couple
    the flow components.

    Batch axes of `mat` and `vec` are broadcast against each other; the spatial
    axes and the last axis must match.

    `flow_precond_` is the in-place version.

    Differentiable with respect to `vec`, through the self-adjoint solve. `mat`
    must not require grad, as for [`sym_solve`][fastfields.torch.sym_solve].

    !!! note "Composed, not fused"
        numpy and cupy call a single fused kernel here. This wrapper
        instead composes `flow_diag` and `sym_solve`, which are
        differentiable, so the gradient flows.
        `fastfields.dlpack.flow_precond` is the fused kernel, but calling
        it directly drops the gradient.

    Parameters
    ----------
    mat : torch.Tensor
        Per-voxel matrix `M` in compact-symmetric packing, shape `(*batch,
        *spatial, ndim*(ndim+1)//2)`. Batch axes are broadcast against
        those of `vec`.
    vec : torch.Tensor
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
    torch.Tensor
        The solution `x`, shape `(*batch, *spatial, ndim)` with the
        broadcast batch shape.
    """
    check_dtype(vec)
    diag = flow_diag(
        vec.shape,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size=voxel_size,
        bound=bound,
        ndim=ndim,
        dtype=vec.dtype,
        device=vec.device,
    )
    return sym_solve(mat, vec, diag)


def flow_forward(
    mat: Tensor,
    vec: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Apply the regularised system: `(M + L) @ vec`.

    Computes `(M + L) @ vec`, where `M` is the per-voxel matrix `mat` and `L`
    is the flow regulariser `L`. This is the operator that
    [`flow_relax`][fastfields.torch.flow_relax] solves. The penalties are
    scalars. A non-zero `shears` or `div` adds the linear-elastic terms, which
    couple the flow components.

    Batch axes of `mat` and `vec` are broadcast against each other; the spatial
    axes and the last axis must match.

    Differentiable with respect to both `mat` and `vec`.

    !!! note "Composed, not fused"
        numpy and cupy call a single fused kernel here. This wrapper
        instead composes `sym_matvec` and `flow_matvec`, which are
        differentiable, so the gradient flows. That costs a temporary and a
        second pass over the data. `fastfields.dlpack.flow_forward` is the
        fused kernel, but calling it directly drops the gradient: the
        forward pass runs and the graph ends there.

    Parameters
    ----------
    mat : torch.Tensor
        Per-voxel matrix `M` in compact-symmetric packing, shape `(*batch,
        *spatial, ndim*(ndim+1)//2)`. Batch axes are broadcast against
        those of `vec`.
    vec : torch.Tensor
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
    torch.Tensor
        `(M + L) @ vec`, shape `(*batch, *spatial, ndim)` with the
        broadcast batch shape.
    """
    check_dtype(vec)
    out = sym_matvec(mat, vec)
    out = out + flow_matvec(
        vec,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size=voxel_size,
        bound=bound,
        ndim=ndim,
    )
    return out


# --- accumulate variants -------------------------------------------------
#
# jitfields' ``_add`` / ``_sub`` (fresh array) and trailing-underscore in-place
# forms, as thin compositions ``inp ± op(...)``. The fresh forms compose the
# autograd flow_matvec, so they stay differentiable; the in-place forms use
# augmented assignment (``add_``/``sub_``), intended for non-autograd use.


def flow_addmatvec(
    inp: Tensor,
    flow: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Add the flow regulariser applied to a flow: `inp + L @ flow`.

    Computes `inp + L @ flow` for the flow regulariser `L`. The penalties are
    scalars. A non-zero `shears` or `div` adds the linear-elastic terms, which
    couple the flow components.

    `inp` is copied, never modified;
    [`flow_addmatvec_`][fastfields.torch.flow_addmatvec_] is the version that
    updates it in place.

    Differentiable with respect to both `inp` and `flow`.

    Parameters
    ----------
    inp : torch.Tensor
        Array the result is accumulated onto, shape `(*batch, *spatial,
        ndim)`.
    flow : torch.Tensor
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
    torch.Tensor
        `inp + L @ flow`, a new tensor the shape of `inp`.
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
    inp: Tensor,
    flow: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Subtract the flow regulariser applied to a flow: `inp - L @ flow`.

    Computes `inp - L @ flow` for the flow regulariser `L`. The penalties are
    scalars. A non-zero `shears` or `div` adds the linear-elastic terms, which
    couple the flow components.

    `inp` is copied, never modified;
    [`flow_submatvec_`][fastfields.torch.flow_submatvec_] is the version that
    updates it in place.

    Differentiable with respect to both `inp` and `flow`.

    Parameters
    ----------
    inp : torch.Tensor
        Array the result is accumulated onto, shape `(*batch, *spatial,
        ndim)`.
    flow : torch.Tensor
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
    torch.Tensor
        `inp - L @ flow`, a new tensor the shape of `inp`.
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
    inp: Tensor,
    flow: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Accumulate the flow regulariser in place: `inp += L @ flow`.

    Updates `inp` with `inp += L @ flow` for the flow regulariser `L`, in one
    kernel and without a temporary for the product. The penalties are scalars.
    A non-zero `shears` or `div` adds the linear-elastic terms, which couple
    the flow components.

    [`flow_addmatvec`][fastfields.torch.flow_addmatvec] is the version that
    returns a new tensor.

    Safe under autograd: the operation is additive in `inp`, so the backward
    pass needs no value that was overwritten. As with `Tensor.add_`, a leaf
    tensor that requires grad cannot be modified in place.

    Parameters
    ----------
    inp : torch.Tensor
        Accumulator, shape `(*batch, *spatial, ndim)`. Updated in place.
    flow : torch.Tensor
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
    torch.Tensor
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
    inp: Tensor,
    flow: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Accumulate the flow regulariser in place: `inp -= L @ flow`.

    Updates `inp` with `inp -= L @ flow` for the flow regulariser `L`, in one
    kernel and without a temporary for the product. The penalties are scalars.
    A non-zero `shears` or `div` adds the linear-elastic terms, which couple
    the flow components.

    [`flow_submatvec`][fastfields.torch.flow_submatvec] is the version that
    returns a new tensor.

    Safe under autograd: the operation is additive in `inp`, so the backward
    pass needs no value that was overwritten. As with `Tensor.add_`, a leaf
    tensor that requires grad cannot be modified in place.

    Parameters
    ----------
    inp : torch.Tensor
        Accumulator, shape `(*batch, *spatial, ndim)`. Updated in place.
    flow : torch.Tensor
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
    torch.Tensor
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
    inp: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Add the diagonal of the flow regulariser: `inp + diag(L)`.

    Computes `inp + diag(L)` for the flow regulariser `L`. The penalties are
    scalars. A non-zero `shears` or `div` adds the linear-elastic terms, which
    couple the flow components.

    `inp` is copied, never modified;
    [`flow_adddiag_`][fastfields.torch.flow_adddiag_] updates it in place.

    Differentiable with respect to `inp`.

    Parameters
    ----------
    inp : torch.Tensor
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
    torch.Tensor
        `inp + diag(L)`, a new tensor the shape of `inp`.
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
    inp: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Subtract the diagonal of the flow regulariser: `inp - diag(L)`.

    Computes `inp - diag(L)` for the flow regulariser `L`. The penalties are
    scalars. A non-zero `shears` or `div` adds the linear-elastic terms, which
    couple the flow components.

    `inp` is copied, never modified;
    [`flow_subdiag_`][fastfields.torch.flow_subdiag_] updates it in place.

    Differentiable with respect to `inp`.

    Parameters
    ----------
    inp : torch.Tensor
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
    torch.Tensor
        `inp - diag(L)`, a new tensor the shape of `inp`.
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
    inp: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Accumulate the flow regulariser's diagonal in place: `inp += diag(L)`.

    Updates `inp` with `inp += diag(L)` for the flow regulariser `L`, in one
    kernel. The penalties are scalars. A non-zero `shears` or `div` adds the
    linear-elastic terms, which couple the flow components.

    [`flow_adddiag`][fastfields.torch.flow_adddiag] returns a new tensor
    instead.

    Safe under autograd: the operation is additive in `inp`. A leaf tensor that
    requires grad cannot be modified in place.

    Parameters
    ----------
    inp : torch.Tensor
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
    torch.Tensor
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
    inp: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Accumulate the flow regulariser's diagonal in place: `inp -= diag(L)`.

    Updates `inp` with `inp -= diag(L)` for the flow regulariser `L`, in one
    kernel. The penalties are scalars. A non-zero `shears` or `div` adds the
    linear-elastic terms, which couple the flow components.

    [`flow_subdiag`][fastfields.torch.flow_subdiag] returns a new tensor
    instead.

    Safe under autograd: the operation is additive in `inp`. A leaf tensor that
    requires grad cannot be modified in place.

    Parameters
    ----------
    inp : torch.Tensor
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
    torch.Tensor
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
    mat: Tensor,
    vec: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Jacobi preconditioner: solve `(M + diag(L)) x = vec` voxel by voxel.

    Solves `(M + diag(L)) x = vec` voxel by voxel, where `M` is the per-voxel
    matrix `mat` and `diag(L)` is the diagonal of the field regulariser `L`. It
    approximates the inverse of the operator of
    [`field_forward`][fastfields.torch.field_forward]. Each penalty is given
    per channel and the regulariser never couples channels.

    Batch axes of `mat` and `vec` are broadcast against each other; the spatial
    axes and the last axis must match.

    `field_precond_` is the in-place version.

    Differentiable with respect to `vec`, through the self-adjoint solve. `mat`
    must not require grad, as for [`sym_solve`][fastfields.torch.sym_solve].

    !!! note "Composed, not fused"
        numpy and cupy call a single fused kernel here. This wrapper
        instead composes `field_diag` and `sym_solve`, which are
        differentiable, so the gradient flows.
        `fastfields.dlpack.field_precond` is the fused kernel, but calling
        it directly drops the gradient.

    Parameters
    ----------
    mat : torch.Tensor
        Per-voxel matrix `M` in compact-symmetric packing, shape `(*batch,
        *spatial, C*(C+1)//2)`. Batch axes are broadcast against those of
        `vec`.
    vec : torch.Tensor
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
    torch.Tensor
        The solution `x`, shape `(*batch, *spatial, C)` with the broadcast
        batch shape.
    """
    check_dtype(vec)
    diag = field_diag(
        vec.shape,
        absolute,
        membrane,
        bending,
        voxel_size=voxel_size,
        bound=bound,
        ndim=ndim,
        dtype=vec.dtype,
        device=vec.device,
    )
    return sym_solve(mat, vec, diag)


def field_forward(
    mat: Tensor,
    vec: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Apply the regularised system: `(M + L) @ vec`.

    Computes `(M + L) @ vec`, where `M` is the per-voxel matrix `mat` and `L`
    is the field regulariser `L`. This is the operator that
    [`field_relax`][fastfields.torch.field_relax] solves. Each penalty is given
    per channel and the regulariser never couples channels.

    Batch axes of `mat` and `vec` are broadcast against each other; the spatial
    axes and the last axis must match.

    Differentiable with respect to both `mat` and `vec`.

    !!! note "Composed, not fused"
        numpy and cupy call a single fused kernel here. This wrapper
        instead composes `sym_matvec` and `field_matvec`, which are
        differentiable, so the gradient flows. That costs a temporary and a
        second pass over the data. `fastfields.dlpack.field_forward` is the
        fused kernel, but calling it directly drops the gradient: the
        forward pass runs and the graph ends there.

    Parameters
    ----------
    mat : torch.Tensor
        Per-voxel matrix `M` in compact-symmetric packing, shape `(*batch,
        *spatial, C*(C+1)//2)`. Batch axes are broadcast against those of
        `vec`.
    vec : torch.Tensor
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
    torch.Tensor
        `(M + L) @ vec`, shape `(*batch, *spatial, C)` with the broadcast
        batch shape.
    """
    check_dtype(vec)
    out = sym_matvec(mat, vec)
    out = out + field_matvec(
        vec,
        absolute,
        membrane,
        bending,
        voxel_size=voxel_size,
        bound=bound,
        ndim=ndim,
    )
    return out


def field_addmatvec(
    inp: Tensor,
    field: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Add the field regulariser applied to a field: `inp + L @ field`.

    Computes `inp + L @ field` for the field regulariser `L`. Each penalty is
    given per channel and the regulariser never couples channels.

    `inp` is copied, never modified;
    [`field_addmatvec_`][fastfields.torch.field_addmatvec_] is the version that
    updates it in place.

    Differentiable with respect to both `inp` and `field`.

    Parameters
    ----------
    inp : torch.Tensor
        Array the result is accumulated onto, shape `(*batch, *spatial,
        C)`.
    field : torch.Tensor
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
    torch.Tensor
        `inp + L @ field`, a new tensor the shape of `inp`.
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
    inp: Tensor,
    field: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Subtract the field regulariser applied to a field: `inp - L @ field`.

    Computes `inp - L @ field` for the field regulariser `L`. Each penalty is
    given per channel and the regulariser never couples channels.

    `inp` is copied, never modified;
    [`field_submatvec_`][fastfields.torch.field_submatvec_] is the version that
    updates it in place.

    Differentiable with respect to both `inp` and `field`.

    Parameters
    ----------
    inp : torch.Tensor
        Array the result is accumulated onto, shape `(*batch, *spatial,
        C)`.
    field : torch.Tensor
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
    torch.Tensor
        `inp - L @ field`, a new tensor the shape of `inp`.
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
    inp: Tensor,
    field: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Accumulate the field regulariser in place: `inp += L @ field`.

    Updates `inp` with `inp += L @ field` for the field regulariser `L`, in one
    kernel and without a temporary for the product. Each penalty is given per
    channel and the regulariser never couples channels.

    [`field_addmatvec`][fastfields.torch.field_addmatvec] is the version that
    returns a new tensor.

    Safe under autograd: the operation is additive in `inp`, so the backward
    pass needs no value that was overwritten. As with `Tensor.add_`, a leaf
    tensor that requires grad cannot be modified in place.

    Parameters
    ----------
    inp : torch.Tensor
        Accumulator, shape `(*batch, *spatial, C)`. Updated in place.
    field : torch.Tensor
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
    torch.Tensor
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
    inp: Tensor,
    field: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Accumulate the field regulariser in place: `inp -= L @ field`.

    Updates `inp` with `inp -= L @ field` for the field regulariser `L`, in one
    kernel and without a temporary for the product. Each penalty is given per
    channel and the regulariser never couples channels.

    [`field_submatvec`][fastfields.torch.field_submatvec] is the version that
    returns a new tensor.

    Safe under autograd: the operation is additive in `inp`, so the backward
    pass needs no value that was overwritten. As with `Tensor.add_`, a leaf
    tensor that requires grad cannot be modified in place.

    Parameters
    ----------
    inp : torch.Tensor
        Accumulator, shape `(*batch, *spatial, C)`. Updated in place.
    field : torch.Tensor
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
    torch.Tensor
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
    inp: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Add the diagonal of the field regulariser: `inp + diag(L)`.

    Computes `inp + diag(L)` for the field regulariser `L`. Each penalty is
    given per channel and the regulariser never couples channels.

    `inp` is copied, never modified;
    [`field_adddiag_`][fastfields.torch.field_adddiag_] updates it in place.

    Differentiable with respect to `inp`.

    Parameters
    ----------
    inp : torch.Tensor
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
    torch.Tensor
        `inp + diag(L)`, a new tensor the shape of `inp`.
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
    inp: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Subtract the diagonal of the field regulariser: `inp - diag(L)`.

    Computes `inp - diag(L)` for the field regulariser `L`. Each penalty is
    given per channel and the regulariser never couples channels.

    `inp` is copied, never modified;
    [`field_subdiag_`][fastfields.torch.field_subdiag_] updates it in place.

    Differentiable with respect to `inp`.

    Parameters
    ----------
    inp : torch.Tensor
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
    torch.Tensor
        `inp - diag(L)`, a new tensor the shape of `inp`.
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
    inp: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Accumulate the field regulariser's diagonal in place: `inp += diag(L)`.

    Updates `inp` with `inp += diag(L)` for the field regulariser `L`, in one
    kernel. Each penalty is given per channel and the regulariser never couples
    channels.

    [`field_adddiag`][fastfields.torch.field_adddiag] returns a new tensor
    instead.

    Safe under autograd: the operation is additive in `inp`. A leaf tensor that
    requires grad cannot be modified in place.

    Parameters
    ----------
    inp : torch.Tensor
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
    torch.Tensor
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
    inp: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Accumulate the field regulariser's diagonal in place: `inp -= diag(L)`.

    Updates `inp` with `inp -= diag(L)` for the field regulariser `L`, in one
    kernel. Each penalty is given per channel and the regulariser never couples
    channels.

    [`field_subdiag`][fastfields.torch.field_subdiag] returns a new tensor
    instead.

    Safe under autograd: the operation is additive in `inp`. A leaf tensor that
    requires grad cannot be modified in place.

    Parameters
    ----------
    inp : torch.Tensor
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
    torch.Tensor
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
# Stencil (kernel) accumulate variants
# ---------------------------------------------------------------------------
#
# The same `inp +/- op(...)` pattern as the *_adddiag / *_subdiag family, for
# the materialised Toeplitz stencil rather than the diagonal. `inp` is shaped
# like `field_kernel` / `flow_kernel`'s output, and the stencil
# depends on no tensor input, so these are differentiable wrt `inp` by
# identity pass-through -- exactly like the diag forms.


def field_addkernel(
    inp: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Add the stencil of the field regulariser: `inp + K`.

    Computes `inp + K`, where `K` is the stencil that
    [`field_kernel`][fastfields.torch.field_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil and is copied, never
    modified; [`field_addkernel_`][fastfields.torch.field_addkernel_] updates
    it in place.

    Differentiable with respect to `inp`.

    Parameters
    ----------
    inp : torch.Tensor
        Array the stencil is accumulated onto, shaped like the output of
        [`field_kernel`][fastfields.torch.field_kernel].
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
    torch.Tensor
        `inp + K`, a new tensor the shape of `inp`.
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
    inp: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Subtract the stencil of the field regulariser: `inp - K`.

    Computes `inp - K`, where `K` is the stencil that
    [`field_kernel`][fastfields.torch.field_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil and is copied, never
    modified; [`field_subkernel_`][fastfields.torch.field_subkernel_] updates
    it in place.

    Differentiable with respect to `inp`.

    Parameters
    ----------
    inp : torch.Tensor
        Array the stencil is accumulated onto, shaped like the output of
        [`field_kernel`][fastfields.torch.field_kernel].
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
    torch.Tensor
        `inp - K`, a new tensor the shape of `inp`.
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
    inp: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Accumulate the field regulariser's stencil in place: `inp += K`.

    Updates `inp` with `inp += K`, where `K` is the stencil that
    [`field_kernel`][fastfields.torch.field_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil.
    [`field_addkernel`][fastfields.torch.field_addkernel] returns a new tensor
    instead.

    Safe under autograd: the operation is additive in `inp`. A leaf tensor that
    requires grad cannot be modified in place.

    Parameters
    ----------
    inp : torch.Tensor
        Accumulator, shaped like the output of
        [`field_kernel`][fastfields.torch.field_kernel]. Updated in place.
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
    torch.Tensor
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
    inp: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Accumulate the field regulariser's stencil in place: `inp -= K`.

    Updates `inp` with `inp -= K`, where `K` is the stencil that
    [`field_kernel`][fastfields.torch.field_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil.
    [`field_subkernel`][fastfields.torch.field_subkernel] returns a new tensor
    instead.

    Safe under autograd: the operation is additive in `inp`. A leaf tensor that
    requires grad cannot be modified in place.

    Parameters
    ----------
    inp : torch.Tensor
        Accumulator, shaped like the output of
        [`field_kernel`][fastfields.torch.field_kernel]. Updated in place.
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
    torch.Tensor
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
    inp: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Add the stencil of the flow regulariser: `inp + K`.

    Computes `inp + K`, where `K` is the stencil that
    [`flow_kernel`][fastfields.torch.flow_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil and is copied, never
    modified; [`flow_addkernel_`][fastfields.torch.flow_addkernel_] updates it
    in place.

    Differentiable with respect to `inp`.

    Parameters
    ----------
    inp : torch.Tensor
        Array the stencil is accumulated onto, shaped like the output of
        [`flow_kernel`][fastfields.torch.flow_kernel].
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
    torch.Tensor
        `inp + K`, a new tensor the shape of `inp`.
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
    inp: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Subtract the stencil of the flow regulariser: `inp - K`.

    Computes `inp - K`, where `K` is the stencil that
    [`flow_kernel`][fastfields.torch.flow_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil and is copied, never
    modified; [`flow_subkernel_`][fastfields.torch.flow_subkernel_] updates it
    in place.

    Differentiable with respect to `inp`.

    Parameters
    ----------
    inp : torch.Tensor
        Array the stencil is accumulated onto, shaped like the output of
        [`flow_kernel`][fastfields.torch.flow_kernel].
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
    torch.Tensor
        `inp - K`, a new tensor the shape of `inp`.
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
    inp: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Accumulate the flow regulariser's stencil in place: `inp += K`.

    Updates `inp` with `inp += K`, where `K` is the stencil that
    [`flow_kernel`][fastfields.torch.flow_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil.
    [`flow_addkernel`][fastfields.torch.flow_addkernel] returns a new tensor
    instead.

    Safe under autograd: the operation is additive in `inp`. A leaf tensor that
    requires grad cannot be modified in place.

    Parameters
    ----------
    inp : torch.Tensor
        Accumulator, shaped like the output of
        [`flow_kernel`][fastfields.torch.flow_kernel]. Updated in place.
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
    torch.Tensor
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
    inp: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Accumulate the flow regulariser's stencil in place: `inp -= K`.

    Updates `inp` with `inp -= K`, where `K` is the stencil that
    [`flow_kernel`][fastfields.torch.flow_kernel] returns for the same
    penalties. `inp` must have the shape of that stencil.
    [`flow_subkernel`][fastfields.torch.flow_subkernel] returns a new tensor
    instead.

    Safe under autograd: the operation is additive in `inp`. A leaf tensor that
    requires grad cannot be modified in place.

    Parameters
    ----------
    inp : torch.Tensor
        Accumulator, shaped like the output of
        [`flow_kernel`][fastfields.torch.flow_kernel]. Updated in place.
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
    torch.Tensor
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
