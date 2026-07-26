"""Backend-dispatching regulariser wrappers (fastfields.any).

Only the operators (``*_matvec``) are dispatched here — they take the field as
their first array argument, so the backend is unambiguous. The ``*_diag``
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
}

for _name, _table in _ALIASES.items():
    globals()[_name] = make_dispatcher(_name, _table)

__all__ = list(_ALIASES)
