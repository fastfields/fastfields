"""Guard: the three ``dt_spline_*`` wrappers return ``(dist, time)``.

fastfields#38. The point-to-spline operators return a 2-tuple, and
``fastfields.cupy`` used to return it as ``(time, dist)`` while
``fastfields.numpy`` and ``fastfields.torch`` returned ``(dist, time)``.
``fastfields.auto`` resolves a backend and returns the result unchanged --
``_util.make_dispatcher`` ends in ``getattr(backend, attr)(*args, **kwargs)``,
so a forwarding dispatcher has nowhere to normalise a *result* tuple -- which
made the meaning of ``fastfields.auto.dt_spline_table(...)``'s return value
depend on which array type the caller passed. The owner's decision was
``(dist, time)`` everywhere, so cupy moved.

The flip is easy to make again, in either direction, because the raw binding
takes its output buffers the *other* way round
(``_ff.dt_spline_*(time, dist, ...)``, the order the C++ hub declares): a
reader who assumes the wrapper mirrors the binding will "fix" the return
statement, and nothing else would complain. ``tests/test_cupy.py`` cannot: it
needs a GPU, and there is none in CI -- which is how the divergence survived
long enough to need an issue in the first place.

So this guard is **static**, for the same reason and by the same means as
``test_reg_family_routing_cupy.py``: it locates the installed
``fastfields/cupy/`` through the namespace package's search locations and
parses the source, importing nothing. ``import fastfields.cupy`` deliberately
fails on a CPU-only ``fastfields-dlpack`` build, which is the build CI has, so
an importing guard would skip in precisely the environment that has to catch
this.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib

import pytest

SPLINE_OPS = ("dt_spline_table", "dt_spline_brent", "dt_spline_gaussnewton")

#: The canonical return order, shared with fastfields.numpy and
#: fastfields.torch (API_CONTRACT.md, "`dt_spline_*` returns `(dist, time)`").
CANONICAL_ORDER = ("dist", "time")

#: The order the raw binding declares its two output buffers in. Deliberately
#: the reverse of the above; that is the whole trap.
BINDING_ORDER = ("time", "dist")


def _cupy_dt_source():
    """Path of the installed ``fastfields/cupy/_dt.py``.

    Found through the ``fastfields`` namespace package's search locations,
    which resolving does not execute any of its portions -- so this works on a
    build where ``import fastfields.cupy`` raises.
    """
    spec = importlib.util.find_spec("fastfields")
    assert spec is not None, "the fastfields namespace is not installed"
    for location in spec.submodule_search_locations:
        candidate = pathlib.Path(location) / "cupy" / "_dt.py"
        if candidate.is_file():
            return candidate
    raise AssertionError(
        "fastfields/cupy/_dt.py not found in any fastfields namespace "
        f"portion: {list(spec.submodule_search_locations)}"
    )


def _function_defs():
    """Every ``def`` at module scope in ``fastfields/cupy/_dt.py``, by name."""
    tree = ast.parse(_cupy_dt_source().read_text())
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _returned_names(node):
    """The names in the single ``return <tuple>`` of ``node``."""
    returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
    assert len(returns) == 1, (
        f"{node.name} has {len(returns)} return statements; this guard "
        "assumes exactly one"
    )
    value = returns[0].value
    assert isinstance(value, ast.Tuple), (
        f"{node.name} does not return a tuple literal any more; update this "
        "guard together with the wrapper"
    )
    names = []
    for element in value.elts:
        assert isinstance(element, ast.Name), (
            f"{node.name} returns a non-name element; update this guard"
        )
        names.append(element.id)
    return tuple(names)


@pytest.mark.parametrize("name", SPLINE_OPS)
def test_wrapper_returns_dist_then_time(name):
    defs = _function_defs()
    assert name in defs, f"fastfields.cupy._dt has no {name!r}"
    assert _returned_names(defs[name]) == CANONICAL_ORDER, (
        f"fastfields.cupy.{name} must return {CANONICAL_ORDER} to match "
        "fastfields.numpy and fastfields.torch (fastfields#38). The binding "
        "takes (time, dist); the wrapper does not."
    )


@pytest.mark.parametrize("name", SPLINE_OPS)
def test_binding_call_keeps_the_hub_argument_order(name):
    """The *call* into the binding stays ``(time, dist, ...)``.

    If someone aligns the call site with the return statement, the wrapper
    still runs -- and silently reports distance as time.
    """
    defs = _function_defs()
    calls = [
        node
        for node in ast.walk(defs[name])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]
    assert len(calls) == 1, f"{name} does not call _ff.{name} exactly once"
    first_two = []
    for arg in calls[0].args[:2]:
        assert isinstance(arg, ast.Name), (
            f"{name}'s binding call no longer passes plain names; update "
            "this guard"
        )
        first_two.append(arg.id)
    assert tuple(first_two) == BINDING_ORDER, (
        f"fastfields.cupy.{name} must pass the binding's output buffers as "
        f"{BINDING_ORDER} -- that is the order fastfields.dlpack declares. "
        "Only the wrapper's *return* order is (dist, time)."
    )


@pytest.mark.parametrize("name", SPLINE_OPS)
def test_docstring_states_the_return_order(name):
    """A caller must not have to read the source to know the order."""
    doc = ast.get_docstring(_function_defs()[name]) or ""
    assert "``(dist, time)``" in doc, (
        f"fastfields.cupy.{name}'s docstring must state the "
        "``(dist, time)`` return order"
    )
    assert "``(time, dist)``" not in doc, (
        f"fastfields.cupy.{name}'s docstring still claims ``(time, dist)``"
    )
