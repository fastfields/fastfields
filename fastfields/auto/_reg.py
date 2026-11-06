"""Dispatchers for the spatial regularisers.

An operator is dispatched when its first argument is an array, which is the
case for `field_matvec` and `flow_matvec`, their accumulate forms, the
relaxation and preconditioning operators, the weighted `*_rls` variants and
the `*_addkernel` and `*_subkernel` accumulators. The builders `field_diag`,
`flow_diag`, `field_kernel` and `flow_kernel` take a shape or a rank and
penalty weights instead, so there is nothing to dispatch on; call them on a
concrete backend.

Not every operator exists on every backend. Those that do not are listed in
`_PARTIAL` with the reason, and a call that lands on a backend without one
raises `NotImplementedError` naming the backends that have it.
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
