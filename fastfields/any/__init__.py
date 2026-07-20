"""fastfields.any: a backend-dispatching interface over the fastfields wrappers.

Every function here dispatches on the **type of the first array argument** and
forwards the call to the matching backend package:

* ``numpy.ndarray`` -> :mod:`fastfields.numpy`
* ``torch.Tensor``  -> :mod:`fastfields.torch`
* ``cupy.ndarray``  -> :mod:`fastfields.cupy`

Backends are imported **lazily**: only the backend for the array type actually
passed is imported, so ``fastfields.any`` works with any subset of the optional
backends installed (only ``fastfields-dlpack`` is a hard dependency). A clear
error is raised for an unknown array type, and a clear ``ImportError`` if the
required backend package (or its own dependency, e.g. torch/cupy) is missing.

Because the underlying wrapper packages use slightly different function names
(e.g. numpy's ``euclidean_distance_transform`` vs torch/cupy's ``dt_euclidean``,
or numpy's ``mesh_distance`` vs torch/cupy's ``dt_mesh``), this module exposes a
single unified name per operation and maps it to each backend's actual
function. Arguments are forwarded unchanged, so each dispatched call shares the
signature of the selected backend's function.
"""

from __future__ import annotations

import importlib

# Enums are backend-independent (they live in fastfields.dlpack).
from fastfields.dlpack import Bound, Spline

__version__ = "0.1.0"

# root module name of an array's type -> backend package to import
_ARRAY_BACKENDS = {
    "numpy": "fastfields.numpy",
    "torch": "fastfields.torch",
    "cupy": "fastfields.cupy",
}

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
    "sym_matvec": {"numpy": "sym_matvec", "torch": "sym_matvec", "cupy": "sym_matvec"},
    "sym_matvec_backward": {
        "numpy": "sym_matvec_backward",
        "cupy": "sym_matvec_backward",
    },
    "sym_solve": {"numpy": "sym_solve", "torch": "sym_solve", "cupy": "sym_solve"},
    "sym_invert": {"numpy": "sym_invert", "torch": "sym_invert", "cupy": "sym_invert"},
    "resample": {"numpy": "resample", "torch": "resample", "cupy": "resample"},
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

_backend_cache: dict[str, object] = {}


def _backend_root(arr) -> str:
    """Return the backend key ('numpy'/'torch'/'cupy') for an array object.

    Detection is by the array type's top-level module name, so we never import
    torch or cupy just to test ``isinstance`` -- only the backend that is
    actually used gets imported.
    """
    root = (type(arr).__module__ or "").split(".")[0]
    if root in _ARRAY_BACKENDS:
        return root
    raise TypeError(
        "fastfields.any could not dispatch on an object of type "
        f"{type(arr).__module__}.{type(arr).__name__}; expected a "
        "numpy.ndarray, torch.Tensor or cupy.ndarray as the first argument."
    )


def _load_backend(root: str):
    backend = _backend_cache.get(root)
    if backend is None:
        try:
            backend = importlib.import_module(_ARRAY_BACKENDS[root])
        except ImportError as exc:
            raise ImportError(
                f"fastfields.any needs the '{root}' backend to handle {root} "
                f"arrays, but importing {_ARRAY_BACKENDS[root]} failed. Install "
                f"it via `pip install fastfields[{root}]`."
            ) from exc
        _backend_cache[root] = backend
    return backend


def _first_array(args, kwargs):
    if args:
        return args[0]
    for value in kwargs.values():
        return value
    raise TypeError(
        "fastfields.any dispatch requires at least one array argument."
    )


def _make_dispatcher(name: str):
    table = _ALIASES[name]

    def _dispatch(*args, **kwargs):
        arr = _first_array(args, kwargs)
        root = _backend_root(arr)
        attr = table.get(root)
        if attr is None:
            raise NotImplementedError(
                f"fastfields.any.{name} is not available for the {root} "
                f"backend (supported by: {', '.join(sorted(table))})."
            )
        backend = _load_backend(root)
        return getattr(backend, attr)(*args, **kwargs)

    _dispatch.__name__ = name
    _dispatch.__qualname__ = name
    _dispatch.__doc__ = (
        f"Dispatch ``{name}`` on the type of the first array argument to the "
        f"corresponding backend function (mapping: {table}). Arguments are "
        f"forwarded unchanged."
    )
    return _dispatch


for _name in _ALIASES:
    globals()[_name] = _make_dispatcher(_name)

__all__ = ["Spline", "Bound", "__version__", *sorted(_ALIASES)]
