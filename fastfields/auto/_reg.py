"""Backend-dispatching regulariser wrappers (fastfields.auto).

Operators that take a field/matrix as their first array argument are dispatched
here (``field_matvec``, ``flow_matvec`` and its ``_add``/``_sub``/in-place
accumulate forms, ``flow_diag`` accumulate forms, ``field_relax`` /
``flow_relax``, ``flow_precond``, ``flow_forward``) — the backend is
unambiguous from that array. The plain ``*_diag`` / ``flow_kernel`` builders
allocate from a *shape* (no array to dispatch on); call them on a concrete
backend (``fastfields.numpy`` / ``.torch`` / ``.cupy``) directly.

The RLS/JRLS-weighted variants (``{field,flow}_matvec_rls``,
``{field,flow}_relax_rls`` and ``{field,flow}_diag_rls``) all take a real
``wgt`` array as an argument -- unlike the plain ``*_diag`` builders,
``field_diag_rls``/``flow_diag_rls`` *are* dispatchable here (their first
argument is ``wgt``, not a shape).

The ``*_addkernel`` / ``*_subkernel`` accumulate forms take the stencil array
as their first argument, so they dispatch here too; the ``*_kernel``
*builders* they accumulate over do not, for the same reason ``*_diag`` does
not -- they are given a rank and penalty weights, not an array.

Not every operator exists on every backend. Those are in `_PARTIAL`
with the reason, and a call that lands on a backend without one raises
``NotImplementedError`` naming the backends that do have it -- rather than
being silently missing from this module, which would make the omission
invisible from here.
"""

from __future__ import annotations

from ._util import make_dispatcher

# Every unified name maps to the identically-named function on each backend.
_DISPATCHED = [
    "field_matvec",
    "field_addmatvec",
    "field_addmatvec_",
    "field_submatvec",
    "field_submatvec_",
    "field_adddiag",
    "field_adddiag_",
    "field_subdiag",
    "field_subdiag_",
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
    "flow_adddiag",
    "flow_adddiag_",
    "flow_subdiag",
    "flow_subdiag_",
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

#: Operators only some backends have, with the backends that do. The
#: dispatcher raises NotImplementedError naming the supported set when a call
#: lands on a backend that is absent here -- which is the point: an operation
#: a backend cannot do should say so, not fail somewhere deeper.
_PARTIAL = {
    # The in-place preconditioner solves. torch has no in-place form: it
    # would have to resolve against autograd's rules for a mutated buffer
    # (see fastfields.torch's docstring), which is a separate piece of work
    # from exposing the op. The out-of-place `*_precond` is on all three.
    "field_precond_": {"numpy": "field_precond_", "cupy": "field_precond_"},
    "flow_precond_": {"numpy": "flow_precond_", "cupy": "flow_precond_"},
    # The CG driver is CPU-only in fastfields-lib (fastfields-lib#34): the
    # hub rejects CUDA arrays, so fastfields.cupy has no field_cg to call.
    # torch could have one on CPU tensors but does not yet.
    "field_cg": {"numpy": "field_cg"},
}

_ALIASES = {
    _name: {"numpy": _name, "torch": _name, "cupy": _name}
    for _name in _DISPATCHED
}
_ALIASES.update(_PARTIAL)

for _name, _table in _ALIASES.items():
    globals()[_name] = make_dispatcher(_name, _table)

__all__ = list(_ALIASES)
