"""Backend-dispatching pushpull wrappers (fastfields.any)."""

from __future__ import annotations

from ._util import make_dispatcher

# unified name -> {backend root: attribute name in that backend package}
_ALIASES = {
    "pull": {"numpy": "pull", "torch": "pull", "cupy": "pull"},
    "push": {"numpy": "push", "torch": "push", "cupy": "push"},
    "count": {"numpy": "count", "torch": "count", "cupy": "count"},
    "grad": {"numpy": "grad", "torch": "grad", "cupy": "grad"},
}

for _name, _table in _ALIASES.items():
    globals()[_name] = make_dispatcher(_name, _table)

__all__ = list(_ALIASES)
