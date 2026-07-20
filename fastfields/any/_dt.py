"""Backend-dispatching distance-transform wrappers (:mod:`fastfields.any`).

Each name here forwards to the matching backend function; the backends use
slightly different names (e.g. numpy's ``euclidean_distance_transform`` vs
torch/cupy's ``dt_euclidean``), which the per-name tables below reconcile.
"""

from __future__ import annotations

from ._util import make_dispatcher

# unified name -> {backend root: attribute name in that backend package}
_ALIASES = {
    "dt_euclidean": {
        "numpy": "euclidean_distance_transform",
        "torch": "dt_euclidean",
        "cupy": "dt_euclidean",
    },
    "dt_l1": {
        "numpy": "l1_distance_transform",
        "torch": "dt_l1",
        "cupy": "dt_l1",
    },
    "spline_distance_table": {
        "numpy": "spline_distance_table",
        "cupy": "dt_spline_table",
    },
    "spline_distance_brent": {
        "numpy": "spline_distance_brent",
        "cupy": "dt_spline_brent",
    },
    "spline_distance_gaussnewton": {
        "numpy": "spline_distance_gaussnewton",
        "cupy": "dt_spline_gaussnewton",
    },
    "mesh_distance": {
        "numpy": "mesh_distance",
        "torch": "dt_mesh",
        "cupy": "dt_mesh",
    },
}

for _name, _table in _ALIASES.items():
    globals()[_name] = make_dispatcher(_name, _table)

__all__ = list(_ALIASES)
