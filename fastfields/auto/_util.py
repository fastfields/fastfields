"""Dispatch machinery for `fastfields.auto`.

Every public wrapper looks at the type of its first array argument and
forwards the call to the matching backend package: `fastfields.numpy`,
`fastfields.torch` or `fastfields.cupy`. A backend is imported only when an
array of its type is passed, so the package works with any subset of them
installed.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

# root module name of an array's type -> backend package to import
_ARRAY_BACKENDS = {
    "numpy": "fastfields.numpy",
    "torch": "fastfields.torch",
    "cupy": "fastfields.cupy",
}

_backend_cache: dict[str, Any] = {}


def _backend_root(arr: Any) -> str:
    """Return the backend key (``'numpy'``/``'torch'``/``'cupy'``) for ``arr``.

    Detection is by the array type's top-level module name, so we never import
    torch or cupy just to test ``isinstance`` -- only the backend that is
    actually used gets imported.

    Parameters
    ----------
    arr : Any
        The first array argument of a dispatched call.

    Returns
    -------
    str
        The backend key.

    Raises
    ------
    TypeError
        If ``arr`` is not a numpy/torch/cupy array.
    """
    root = (type(arr).__module__ or "").split(".")[0]
    if root in _ARRAY_BACKENDS:
        return root
    raise TypeError(
        "fastfields.auto could not dispatch on an object of type "
        f"{type(arr).__module__}.{type(arr).__name__}; expected a "
        "numpy.ndarray, torch.Tensor or cupy.ndarray as the first argument."
    )


def _load_backend(root: str) -> Any:
    """Import (and cache) the backend package for ``root``.

    Parameters
    ----------
    root : str
        A backend key from `_ARRAY_BACKENDS`.

    Returns
    -------
    module
        The imported backend package.

    Raises
    ------
    ImportError
        If the backend package (or its own dependency) cannot be imported.
    """
    backend = _backend_cache.get(root)
    if backend is None:
        try:
            backend = importlib.import_module(_ARRAY_BACKENDS[root])
        except ImportError as exc:
            raise ImportError(
                f"fastfields.auto needs the '{root}' backend to handle {root} "
                f"arrays, but importing {_ARRAY_BACKENDS[root]} failed. "
                f"Install it via `pip install fastfields[{root}]`."
            ) from exc
        _backend_cache[root] = backend
    return backend


def _first_array(args: tuple, kwargs: dict) -> Any:
    """Return the first positional (or, failing that, keyword) argument.

    Parameters
    ----------
    args : tuple
        Positional arguments of the dispatched call.
    kwargs : dict
        Keyword arguments of the dispatched call.

    Returns
    -------
    Any
        The argument used to select the backend.

    Raises
    ------
    TypeError
        If no arguments were passed.
    """
    if args:
        return args[0]
    for value in kwargs.values():
        return value
    raise TypeError(
        "fastfields.auto dispatch requires at least one array argument."
    )


def make_dispatcher(name: str, table: dict[str, str]) -> Callable[..., Any]:
    """Build a dispatcher forwarding ``name`` to the right backend function.

    Parameters
    ----------
    name : str
        The unified operation name exposed by
        [`fastfields.auto`][fastfields.auto].
    table : dict of {str: str}
        Mapping ``{backend_root: attribute_name}`` giving the backend function
        that implements ``name`` in each backend package.

    Returns
    -------
    callable
        A function that dispatches on the type of its first array argument and
        forwards all arguments unchanged to the selected backend function.
    """

    def _dispatch(*args: Any, **kwargs: Any) -> Any:
        arr = _first_array(args, kwargs)
        root = _backend_root(arr)
        attr = table.get(root)
        if attr is None:
            raise NotImplementedError(
                f"fastfields.auto.{name} is not available for the {root} "
                f"backend (supported by: {', '.join(sorted(table))})."
            )
        backend = _load_backend(root)
        fn = getattr(backend, attr, None)
        if fn is None:
            # The table said this backend has the op and it does not. A blind
            # `getattr(backend, attr)` would surface that as a bare
            # AttributeError on a module the caller never named, which says
            # nothing about what to do. Name all three things: the unified op,
            # the backend that lacks it, and the likely cause.
            #
            # Since the five namespaces were merged into one distribution,
            # auto and the wrapper it dispatched to always ship together, so
            # the remaining way to get here is a stale or partial install --
            # typically a leftover copy of one of the retired
            # fastfields-numpy / -torch / -cupy distributions shadowing this
            # one in the same namespace. Reinstalling resolves it.
            #
            # NotImplementedError rather than AttributeError, to match the
            # adjacent "this backend is not in the table" raise: from a
            # caller's point of view both mean the same thing.
            raise NotImplementedError(
                f"fastfields.auto.{name} dispatched to the {root} backend, "
                f"but {_ARRAY_BACKENDS[root]} has no {attr!r}. These ship in "
                f"the same distribution, so they cannot normally disagree; "
                f"the usual cause is a stale install of a retired "
                f"fastfields-{root} distribution shadowing it in the "
                f"fastfields namespace. Reinstall "
                f"(`pip uninstall fastfields-{root} && pip install -U "
                f"fastfields[{root}]`). Operations deliberately absent from a "
                f"backend are listed in the fastfields.auto docstring."
            )
        return fn(*args, **kwargs)

    _dispatch.__name__ = name
    _dispatch.__qualname__ = name
    _dispatch.__doc__ = (
        f"Dispatch ``{name}`` on the type of the first array argument to the "
        f"corresponding backend function (mapping: {table}). Arguments are "
        f"forwarded unchanged."
    )
    return _dispatch


def dlpack_backend() -> "str | None":
    """Report the compute backend of the installed `fastfields-dlpack`.

    This namespace works with whatever backend is installed and never insists
    on a particular one; this is the place to ask which one that is, without
    knowing which wrapper to interrogate. It is a function and not a module
    attribute so that importing `fastfields.auto` does not have to import the
    compiled bindings.

    Returns
    -------
    str or None
        `"cpu"`, `"cu126"` and so on for a labelled build. `None` if the build
        carries no label (an unversioned source checkout), or if
        `fastfields.dlpack` cannot be imported at all.
    """
    try:
        import fastfields.dlpack as _ff
    except ImportError:
        return None
    return getattr(_ff, "backend", None)
