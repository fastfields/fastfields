"""Backend-dispatching distance-transform wrappers (:mod:`fastfields.auto`).

Each name here forwards to the matching backend function; every backend now
exposes the canonical ``dt_*`` names, so the per-name tables below map each
unified name onto the same attribute in each backend package.

The three ``dt_spline_*`` operators return a 2-tuple, and every backend
returns it as ``(dist, time)``. ``fastfields.cupy`` used to return
``(time, dist)``, which made the meaning of a dispatched result depend on
which array type was passed -- a dispatcher that forwards arguments unchanged
has nowhere to normalise a *result*. Resolved in favour of ``(dist, time)``
(fastfields#38). The three names are in ``tests/test_conformance.py``'s
canonical list now; their absence from it is how the divergence got in.
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
    "dt_euclidean_": {
        "numpy": "dt_euclidean_",
        "torch": "dt_euclidean_",
        "cupy": "dt_euclidean_",
    },
    "dt_l1": {
        "numpy": "dt_l1",
        "torch": "dt_l1",
        "cupy": "dt_l1",
    },
    "dt_l1_": {
        "numpy": "dt_l1_",
        "torch": "dt_l1_",
        "cupy": "dt_l1_",
    },
    # torch gained these in fastfields-torch#35; the table lists it, and an
    # older installed fastfields-torch raises the "upgrade it" message from
    # _util rather than a bare AttributeError.
    "dt_spline_table": {
        "numpy": "dt_spline_table",
        "torch": "dt_spline_table",
        "cupy": "dt_spline_table",
    },
    "dt_spline_brent": {
        "numpy": "dt_spline_brent",
        "torch": "dt_spline_brent",
        "cupy": "dt_spline_brent",
    },
    "dt_spline_gaussnewton": {
        "numpy": "dt_spline_gaussnewton",
        "torch": "dt_spline_gaussnewton",
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
