"""Backend-dispatching resampling / spline-coeff wrappers (fastfields.auto)."""

from __future__ import annotations

from ._util import make_dispatcher

# unified name -> {backend root: attribute name in that backend package}
_ALIASES = {
    "resample": {
        "numpy": "resample",
        "torch": "resample",
        "cupy": "resample",
    },
    "restriction": {
        "numpy": "restriction",
        "torch": "restriction",
        "cupy": "restriction",
    },
    "spline_coeff": {
        "numpy": "spline_coeff",
        "torch": "spline_coeff",
        "cupy": "spline_coeff",
    },
}

for _name, _table in _ALIASES.items():
    globals()[_name] = make_dispatcher(_name, _table)

__all__ = list(_ALIASES)
