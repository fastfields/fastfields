"""Backend-dispatching compact-symmetric ops (:mod:`fastfields.any`)."""

from __future__ import annotations

from ._util import make_dispatcher

# unified name -> {backend root: attribute name in that backend package}
_ALIASES = {
    "sym_matvec": {
        "numpy": "sym_matvec",
        "torch": "sym_matvec",
        "cupy": "sym_matvec",
    },
    "sym_matvec_backward": {
        "numpy": "sym_matvec_backward",
        "cupy": "sym_matvec_backward",
    },
    "sym_solve": {
        "numpy": "sym_solve",
        "torch": "sym_solve",
        "cupy": "sym_solve",
    },
    "sym_invert": {
        "numpy": "sym_invert",
        "torch": "sym_invert",
        "cupy": "sym_invert",
    },
}

for _name, _table in _ALIASES.items():
    globals()[_name] = make_dispatcher(_name, _table)

__all__ = list(_ALIASES)
