"""What ``fastfields.auto`` exposes, and how it fails.

``fastfields.auto`` is a dispatcher: it picks a backend from the type of the
first array argument and forwards everything unchanged. That makes its surface
exactly as good as its name tables, and it used to forward blindly --
``getattr(backend, attr)(*args, **kwargs)`` -- so a name table that promised an
operation the installed wrapper did not have produced a bare
``AttributeError`` mentioning a module the caller never named. A mixed install
is ordinary here: the five distributions are versioned and released
independently.

These tests pin down three things:

1. every dispatched name resolves on every backend its table claims (so the
   tables cannot drift from the wrappers unnoticed);
2. a backend that genuinely lacks an operation produces ``NotImplementedError``
   naming the backends that do have it, not a bare ``AttributeError``;
3. the operations that are deliberately *not* dispatched -- shape-taking
   builders and the autograd-only adjoints -- are documented in the module
   docstring, so their absence is readable rather than inferred.

numpy is a hard test dependency; torch is imported via ``importorskip`` and
cupy skips (it needs a CUDA device).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import fastfields.auto as ff
from fastfields.auto import _dt, _pushpull, _reg, _resample, _sym, _util

_MODULES = (_dt, _sym, _resample, _pushpull, _reg)


def _all_tables():
    """``{unified name: {backend root: attribute}}`` across every category."""
    tables = {}
    for mod in _MODULES:
        tables.update(mod._ALIASES)
    return tables


def _backend_or_skip(root: str):
    try:
        return importlib.import_module(_util._ARRAY_BACKENDS[root])
    except Exception:  # ImportError, or cupy raising with no GPU
        pytest.skip(f"backend {root} not available")


# --------------------------------------------------------------------------- #
# 1. the tables agree with the wrappers                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,root,attr",
    sorted(
        (name, root, attr)
        for name, table in _all_tables().items()
        for root, attr in table.items()
    ),
)
def test_table_entry_resolves_on_its_backend(name, root, attr):
    backend = _backend_or_skip(root)
    assert hasattr(backend, attr), (
        f"fastfields.auto.{name} maps the {root} backend to "
        f"{_util._ARRAY_BACKENDS[root]}.{attr}, which does not exist. Either "
        f"the wrapper dropped it, or the table is wrong."
    )


def test_every_dispatched_name_is_exported():
    missing = sorted(n for n in _all_tables() if n not in ff.__all__)
    assert not missing, (
        f"dispatchers not in fastfields.auto.__all__: {missing}"
    )


def test_dunder_all_resolves():
    missing = sorted(n for n in ff.__all__ if not hasattr(ff, n))
    assert not missing, f"__all__ names that do not exist: {missing}"


# --------------------------------------------------------------------------- #
# 2. failures are intelligible                                                #
# --------------------------------------------------------------------------- #


def test_backend_without_the_op_raises_not_implemented():
    """A partial table must name the backends that do have the op."""
    torch = pytest.importorskip("torch")
    # field_cg is CPU-only in fastfields-lib, so only the numpy backend has
    # it; dispatching a torch tensor must say so.
    with pytest.raises(NotImplementedError) as exc:
        ff.field_cg(
            torch.zeros(4, 1, dtype=torch.float64),
            torch.ones(4, 1, dtype=torch.float64),
            torch.ones(4, 1, dtype=torch.float64),
        )
    msg = str(exc.value)
    assert "field_cg" in msg
    assert "torch" in msg
    assert "numpy" in msg, (
        "the error must name the backends that *do* support the op: " + msg
    )


def test_missing_attribute_is_not_a_bare_attribute_error():
    """A table that over-promises must not surface as an AttributeError.

    Simulates the mixed install (``fastfields.auto`` newer than the installed
    wrapper) by dispatching a name the backend module does not define.
    """
    dispatch = _util.make_dispatcher(
        "not_a_real_op", {"numpy": "definitely_not_defined"}
    )
    with pytest.raises(NotImplementedError) as exc:
        dispatch(np.zeros((2, 1)))
    msg = str(exc.value)
    assert "definitely_not_defined" in msg
    assert "fastfields.numpy" in msg
    assert "upgrade" in msg.lower() or "pip install" in msg, (
        "the error should say how to fix it: " + msg
    )


def test_unknown_array_type_still_raises_type_error():
    with pytest.raises(TypeError, match="could not dispatch"):
        ff.sym_matvec([1.0, 2.0], [1.0])


# --------------------------------------------------------------------------- #
# 3. the non-dispatchable operations are documented                           #
# --------------------------------------------------------------------------- #

#: Operations that cannot be dispatched, and must be explained in the
#: module docstring rather than merely missing from it.
NOT_DISPATCHABLE = [
    # allocate from a shape: there is no array to dispatch on
    "field_diag",
    "flow_diag",
    "field_kernel",
    "flow_kernel",
    # adjoints wired into torch's autograd; no uniform surface to dispatch
    "pull_backward",
    "push_backward",
    "count_backward",
    "grad_backward",
]


@pytest.mark.parametrize("name", NOT_DISPATCHABLE)
def test_not_dispatchable_is_absent_and_explained(name):
    assert name not in ff.__all__, (
        f"{name} is now dispatched; drop it from NOT_DISPATCHABLE here"
    )
    assert name in (ff.__doc__ or ""), (
        f"{name} cannot be dispatched, but the fastfields.auto docstring does "
        "not say so -- an undocumented absence is indistinguishable from an "
        "oversight"
    )


@pytest.mark.parametrize("name", sorted(_reg._PARTIAL))
def test_partial_support_is_explained(name):
    """An op only some backends have must be dispatched, not omitted."""
    assert name in ff.__all__
    table = _reg._PARTIAL[name]
    assert table, f"{name} has an empty backend table"
    assert set(table) < {"numpy", "torch", "cupy"}, (
        f"{name} is in _PARTIAL but supported everywhere; move it to "
        "_DISPATCHED"
    )
