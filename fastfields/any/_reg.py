"""Backend-dispatching regulariser wrappers (fastfields.any).

Operators that take a field/matrix as their first array argument are dispatched
here (``field_matvec``, ``flow_matvec``, ``flow_relax``, ``flow_precond``,
``flow_forward``) — the backend is unambiguous from that array. The ``*_diag``
preconditioners allocate from a *shape* (no array to dispatch on); call them on
a concrete backend (``fastfields.numpy`` / ``.torch`` / ``.cupy``) directly.
"""

from __future__ import annotations

from ._util import make_dispatcher

_ALIASES = {
    "field_matvec": {
        "numpy": "field_matvec",
        "torch": "field_matvec",
        "cupy": "field_matvec",
    },
    "flow_matvec": {
        "numpy": "flow_matvec",
        "torch": "flow_matvec",
        "cupy": "flow_matvec",
    },
    "flow_relax": {
        "numpy": "flow_relax",
        "torch": "flow_relax",
        "cupy": "flow_relax",
    },
    "flow_precond": {
        "numpy": "flow_precond",
        "torch": "flow_precond",
        "cupy": "flow_precond",
    },
    "flow_forward": {
        "numpy": "flow_forward",
        "torch": "flow_forward",
        "cupy": "flow_forward",
    },
}

for _name, _table in _ALIASES.items():
    globals()[_name] = make_dispatcher(_name, _table)

__all__ = list(_ALIASES)
