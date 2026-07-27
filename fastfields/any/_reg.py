"""Backend-dispatching regulariser wrappers (fastfields.any).

Operators that take a field/matrix as their first array argument are dispatched
here (``field_matvec``, ``flow_matvec`` and its ``_add``/``_sub``/in-place
accumulate forms, ``flow_diag`` accumulate forms, ``flow_relax``,
``flow_precond``, ``flow_forward``) — the backend is unambiguous from that
array. The plain ``*_diag`` / ``flow_kernel`` builders allocate from a *shape*
(no array to dispatch on); call them on a concrete backend
(``fastfields.numpy`` / ``.torch`` / ``.cupy``) directly.
"""

from __future__ import annotations

from ._util import make_dispatcher

# Every unified name maps to the identically-named function on each backend.
_DISPATCHED = [
    "field_matvec",
    "field_matvec_add",
    "field_matvec_add_",
    "field_matvec_sub",
    "field_matvec_sub_",
    "field_diag_add",
    "field_diag_add_",
    "field_diag_sub",
    "field_diag_sub_",
    "field_precond",
    "field_forward",
    "flow_matvec",
    "flow_matvec_add",
    "flow_matvec_add_",
    "flow_matvec_sub",
    "flow_matvec_sub_",
    "flow_diag_add",
    "flow_diag_add_",
    "flow_diag_sub",
    "flow_diag_sub_",
    "flow_relax",
    "flow_precond",
    "flow_forward",
]

_ALIASES = {
    _name: {"numpy": _name, "torch": _name, "cupy": _name}
    for _name in _DISPATCHED
}

for _name, _table in _ALIASES.items():
    globals()[_name] = make_dispatcher(_name, _table)

__all__ = list(_ALIASES)
