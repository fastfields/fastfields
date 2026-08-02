"""Backend-dispatching distance-transform wrappers (:mod:`fastfields.auto`).

Each name here forwards to the matching backend function; every backend now
exposes the canonical ``dt_*`` names, so the per-name tables below map each
unified name onto the same attribute in each backend package.
"""

from __future__ import annotations

from ._util import make_dispatcher

# unified name -> {backend root: attribute name in that backend package}
_ALIASES = {
    "dt_euclidean": {
        "numpy": "dt_euclidean",
        "torch": "dt_euclidean",
        "cupy": "dt_euclidean",
    },
    "dt_l1": {
        "numpy": "dt_l1",
        "torch": "dt_l1",
        "cupy": "dt_l1",
    },
    "dt_spline_table": {
        "numpy": "dt_spline_table",
        "cupy": "dt_spline_table",
    },
    "dt_spline_brent": {
        "numpy": "dt_spline_brent",
        "cupy": "dt_spline_brent",
    },
    "dt_spline_gaussnewton": {
        "numpy": "dt_spline_gaussnewton",
        "cupy": "dt_spline_gaussnewton",
    },
    "dt_mesh": {
        "numpy": "dt_mesh",
        "torch": "dt_mesh",
        "cupy": "dt_mesh",
    },
}

for _name, _table in _ALIASES.items():
    globals()[_name] = make_dispatcher(_name, _table)

__all__ = list(_ALIASES)
