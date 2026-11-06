"""Static cross-backend *signature* conformance (``API_CONTRACT.md``).

``tests/test_conformance.py`` checks that every backend exposes the canonical
operation **names**. That is not enough, and the gap is not hypothetical:
``fastfields.cupy.dt_spline_brent`` was spelled
``(loc, coeff, max_iter, tol, step, spline=3, bound=3)`` while numpy and torch
spelled it ``(loc, coeff, *, max_iter=64, tol=1e-6, step=0.1, order=3,
bound="dct2")``, and the name check passed throughout. ``fastfields.auto``
ends in ``getattr(backend, attr)(*args, **kwargs)`` -- it forwards arguments
unchanged and has nowhere to translate a parameter name -- so a divergence
like that one is invisible until a call written against one backend raises
``TypeError`` on another. This module is the gate for that class of bug: the
parameter names, their kinds (positional-or-keyword vs keyword-only), their
order and their defaults must agree on every backend that has the operation.

Why this is **static**, and must stay static
--------------------------------------------
The same reason as ``test_dt_spline_return_order_cupy.py`` and
``test_reg_family_routing_cupy.py``: ``import fastfields.cupy`` deliberately
refuses a CPU-only ``fastfields-dlpack`` build, which is the build CI has, and
every runtime cupy parametrisation therefore *skips* -- in exactly the
environment that has to catch a cupy divergence. A skipping test catches
nothing. So this module imports none of numpy, torch, cupy or
``fastfields.dlpack``: it locates each namespace portion through the
``fastfields`` namespace package's search locations and parses the sources with
``ast``. It cannot skip, and it reports all three backends on a CPU runner
with no GPU and no torch.

What is deliberately *allowed* to differ
----------------------------------------
Two things, both narrow and both named below rather than inferred:

* ``dtype`` defaults (:data:`BACKEND_NATIVE_DEFAULT_PARAMS`). The builders
  ``{field,flow}_{diag,kernel}`` and ``{field,flow}_diag_rls`` allocate their
  own output, so they take a ``dtype``. Its default cannot be spelled portably
  -- ``np.float64``, ``torch.float64`` and (because ``fastfields.cupy``
  imports cupy lazily and so cannot name ``cupy.float64`` at module scope)
  ``None``, resolved to ``cp.float64`` in the body. All three mean float64.
  The parameter's name, kind and position are still checked; only the default
  expression is exempt.
* ``device`` on torch (:data:`BACKEND_ONLY_PARAMS`). A trailing keyword-only
  parameter with a default, which ``API_CONTRACT.md`` explicitly permits a
  backend to add ("a backend may _add_ trailing keyword parameters, but never
  rename, drop, or reorder a canonical parameter"). numpy and cupy have no
  device to choose.

Anything else that differs is a failure, and adding to either list is a
deliberate, reviewable act.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib

import pytest

BACKENDS = ("numpy", "cupy", "torch")

#: Parameters whose *default* is backend-native and so exempt from comparison.
#: Name, kind and position are still compared. See the module docstring.
BACKEND_NATIVE_DEFAULT_PARAMS = frozenset({"dtype"})

#: ``(backend, parameter)`` pairs one backend may add. Each must be
#: keyword-only and carry a default, so a call written against the shared
#: signature keeps working everywhere.
BACKEND_ONLY_PARAMS = frozenset({("torch", "device")})

#: Operations that exist on a subset of the backends, with the exact subset.
#: Asserted exactly, in both directions: a backend that loses one of these
#: fails, and a backend that *gains* one fails too, so the documented absence
#: in ``API_CONTRACT.md`` cannot go stale. Reasons live in
#: ``API_CONTRACT.md`` ("Backend-specific extensions") and in each package's
#: docstring.
PARTIAL_OPS = {
    # The CG driver is CPU-only in fastfields-lib (fastfields-lib#34), so
    # cupy has none; torch could have one on CPU tensors but does not yet.
    "field_cg": frozenset({"numpy"}),
    "CGInfo": frozenset({"numpy"}),
    # In-place preconditioner solves: torch has to resolve autograd's rules
    # for a mutated buffer first.
    "field_precond_": frozenset({"numpy", "cupy"}),
    "flow_precond_": frozenset({"numpy", "cupy"}),
    # On torch the matvec adjoint is wired into sym_matvec's
    # autograd.Function rather than exposed as a public function.
    "sym_matvec_backward": frozenset({"numpy", "cupy"}),
    # Pure-integer packing helper; no cupy/torch equivalent.
    "sym_channels_from_packed": frozenset({"numpy"}),
    # cupy-only: reports the stream every cupy wrapper forwards.
    "current_stream_ptr": frozenset({"cupy"}),
}

#: Names in a backend's ``__all__`` that are not operations.
NON_OPS = frozenset({"__version__", "Spline", "Bound"})

#: Ops no ``fastfields.auto`` dispatcher forwards, with why. ``auto``
#: dispatches on the type of the first array argument, so an operation with no
#: array argument cannot be routed.
AUTO_EXEMPT = {
    "field_diag": "allocates from a shape; no array to dispatch on",
    "field_kernel": "allocates from a rank; no array to dispatch on",
    "flow_diag": "allocates from a shape; no array to dispatch on",
    "flow_kernel": "allocates from a rank; no array to dispatch on",
    "sym_channels_from_packed": "takes an int, not an array",
    "current_stream_ptr": "takes no argument at all",
    "CGInfo": "a result record, not an operation",
}


# --------------------------------------------------------------------------- #
# Locating and parsing the installed sources (no imports)                     #
# --------------------------------------------------------------------------- #


def _portion(subpackage: str) -> pathlib.Path:
    """Directory of the installed ``fastfields/<subpackage>/``.

    Found through the ``fastfields`` namespace package's search locations.
    Resolving those executes none of its portions, so this works on a build
    where ``import fastfields.cupy`` raises.
    """
    spec = importlib.util.find_spec("fastfields")
    assert spec is not None, "the fastfields namespace is not installed"
    locations = list(spec.submodule_search_locations)
    for location in locations:
        candidate = pathlib.Path(location) / subpackage
        if (candidate / "__init__.py").is_file():
            return candidate
    raise AssertionError(
        f"fastfields/{subpackage}/ not found in any fastfields namespace "
        f"portion: {locations}"
    )


def _reexported(init: pathlib.Path) -> "tuple[list, dict]":
    """``(__all__, {name: relative submodule})`` of a package ``__init__``."""
    tree = ast.parse(init.read_text())
    names: list = []
    origin: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    names = [
                        element.value
                        for element in node.value.elts
                        if isinstance(element, ast.Constant)
                    ]
        elif isinstance(node, ast.ImportFrom) and node.level == 1:
            for alias in node.names:
                origin[alias.asname or alias.name] = node.module
    return names, origin


def _defs(package: pathlib.Path) -> dict:
    """``{name: ast node}`` for every module-scope def/class in ``_*.py``."""
    found: dict = {}
    for source in sorted(package.glob("_*.py")):
        for node in ast.parse(source.read_text()).body:
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                found.setdefault(node.name, node)
    return found


def _signature(node: ast.FunctionDef) -> "list[tuple]":
    """``[(name, kind, default_source), ...]`` in declaration order.

    ``kind`` is one of ``"positional"``, ``"keyword-only"``, ``"*args"`` or
    ``"**kwargs"``; ``default_source`` is the unparsed default expression, or
    ``None`` where there is no default.
    """
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    pad = len(positional) - len(args.defaults)
    out = []
    for index, arg in enumerate(positional):
        default = (
            ast.unparse(args.defaults[index - pad]) if index >= pad else None
        )
        out.append((arg.arg, "positional", default))
    if args.vararg:
        out.append((args.vararg.arg, "*args", None))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        out.append(
            (
                arg.arg,
                "keyword-only",
                ast.unparse(default) if default is not None else None,
            )
        )
    if args.kwarg:
        out.append((args.kwarg.arg, "**kwargs", None))
    return out


def _surface() -> dict:
    """``{backend: {op: [(name, kind, default), ...] or None}}``.

    ``None`` means the name is exported but is not a module-scope function in
    that package (a re-exported enum, or a class).
    """
    surface: dict = {}
    for backend in BACKENDS:
        package = _portion(backend)
        names, _origin = _reexported(package / "__init__.py")
        defs = _defs(package)
        ops = {}
        for name in names:
            if name in NON_OPS:
                continue
            node = defs.get(name)
            ops[name] = (
                _signature(node) if isinstance(node, ast.FunctionDef) else None
            )
        surface[backend] = ops
    return surface


def _auto_tables() -> dict:
    """``{op: {backend: attribute}}`` for every fastfields.auto dispatcher.

    Reads the ``_ALIASES`` / ``_DISPATCHED`` / ``_PARTIAL`` literals of
    ``fastfields/auto/_*.py`` without importing anything.
    """
    tables: dict = {}
    package = _portion("auto")
    for source in sorted(package.glob("_*.py")):
        literals: dict = {}
        for node in ast.parse(source.read_text()).body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id in ("_ALIASES", "_PARTIAL") and isinstance(
                    node.value, ast.Dict
                ):
                    for key, value in zip(node.value.keys, node.value.values):
                        if not isinstance(key, ast.Constant):
                            continue
                        if not isinstance(value, ast.Dict):
                            continue
                        literals[key.value] = {
                            inner_key.value: inner_value.value
                            for inner_key, inner_value in zip(
                                value.keys, value.values
                            )
                            if isinstance(inner_key, ast.Constant)
                            and isinstance(inner_value, ast.Constant)
                        }
                elif target.id == "_DISPATCHED" and isinstance(
                    node.value, (ast.List, ast.Tuple)
                ):
                    for element in node.value.elts:
                        if isinstance(element, ast.Constant):
                            literals.setdefault(
                                element.value,
                                {b: element.value for b in BACKENDS},
                            )
        tables.update(literals)
    return tables


SURFACE = _surface()
AUTO_TABLES = _auto_tables()

#: Every operation name any backend exports.
ALL_OPS = sorted(set().union(*(set(ops) for ops in SURFACE.values())))

#: Operations every backend exports as a module-scope function -- the set the
#: signature comparison applies to.
SHARED_OPS = sorted(
    op
    for op in ALL_OPS
    if all(SURFACE[b].get(op) is not None for b in BACKENDS)
)


def test_the_parser_found_a_real_surface():
    """Guard against a vacuous pass from a parse that found nothing."""
    for backend in BACKENDS:
        functions = [
            op for op, sig in SURFACE[backend].items() if sig is not None
        ]
        assert len(functions) > 40, (
            f"only {len(functions)} parsed functions in fastfields.{backend}; "
            "the parser is not seeing the installed sources"
        )
    assert len(SHARED_OPS) > 40, (
        f"only {len(SHARED_OPS)} operations are shared by all three "
        "backends; the comparison below would be nearly empty"
    )
    assert len(AUTO_TABLES) > 40, (
        f"only {len(AUTO_TABLES)} fastfields.auto dispatchers parsed"
    )


# --------------------------------------------------------------------------- #
# 1. Signatures agree across backends                                        #
# --------------------------------------------------------------------------- #


def _comparable(backend: str, signature: "list[tuple]") -> "list[tuple]":
    """Drop a backend's permitted extras and its native ``dtype`` default."""
    out = []
    for name, kind, default in signature:
        if (backend, name) in BACKEND_ONLY_PARAMS:
            continue
        if name in BACKEND_NATIVE_DEFAULT_PARAMS:
            default = "<backend-native>"
        out.append((name, kind, default))
    return out


def _render(signature: "list[tuple]") -> str:
    parts = []
    seen_keyword_only = False
    for name, kind, default in signature:
        if kind == "*args":
            parts.append("*" + name)
            seen_keyword_only = True
            continue
        if kind == "**kwargs":
            parts.append("**" + name)
            continue
        if kind == "keyword-only" and not seen_keyword_only:
            parts.append("*")
            seen_keyword_only = True
        parts.append(name if default is None else f"{name}={default}")
    return "(" + ", ".join(parts) + ")"


@pytest.mark.parametrize("op", SHARED_OPS)
def test_signature_agrees_across_backends(op):
    rendered = {
        backend: _comparable(backend, SURFACE[backend][op])
        for backend in BACKENDS
    }
    distinct = {tuple(signature) for signature in rendered.values()}
    assert len(distinct) == 1, (
        f"{op} does not have the same signature on every backend "
        "(API_CONTRACT.md: a backend may add trailing keyword parameters, "
        "but never rename, drop or reorder a canonical parameter -- and "
        "fastfields.auto forwards arguments unchanged, so a rename here is a "
        "TypeError for one backend only):\n"
        + "\n".join(
            f"  fastfields.{backend}.{op}{_render(signature)}"
            for backend, signature in rendered.items()
        )
    )


@pytest.mark.parametrize("op", SHARED_OPS)
def test_a_backends_extra_parameters_are_optional_and_keyword_only(op):
    """An added parameter must not change how the shared call works."""
    for backend in BACKENDS:
        for name, kind, default in SURFACE[backend][op]:
            if (backend, name) not in BACKEND_ONLY_PARAMS:
                continue
            assert kind == "keyword-only" and default is not None, (
                f"fastfields.{backend}.{op}'s extra parameter {name!r} must "
                "be keyword-only with a default, so a call written against "
                f"the shared signature still works (it is {kind} with "
                f"default {default!r})"
            )


def test_no_undeclared_backend_only_parameter():
    """A parameter only one backend has must be in ``BACKEND_ONLY_PARAMS``."""
    undeclared = []
    for op in SHARED_OPS:
        per_backend = {
            backend: {name for name, _, _ in SURFACE[backend][op]}
            for backend in BACKENDS
        }
        shared = set.intersection(*per_backend.values())
        for backend, names in per_backend.items():
            for name in sorted(names - shared):
                if (backend, name) not in BACKEND_ONLY_PARAMS:
                    undeclared.append(f"{backend}.{op}({name}=...)")
    assert not undeclared, (
        "parameter(s) present on one backend only and not declared in "
        f"BACKEND_ONLY_PARAMS: {undeclared}. Either add the parameter to the "
        "other backends, or declare it here with the reason."
    )


# --------------------------------------------------------------------------- #
# 2. The dt_spline_* family, spelled out                                     #
# --------------------------------------------------------------------------- #
#
# The divergence this module was written for. Pinned by value as well as by
# cross-backend agreement, so "align all three the wrong way" also fails.

DT_SPLINE_SIGNATURES = {
    "dt_spline_table": [
        ("loc", "positional", None),
        ("coeff", "positional", None),
        ("times", "positional", None),
        ("order", "keyword-only", "3"),
        ("bound", "keyword-only", "'dct2'"),
    ],
    "dt_spline_brent": [
        ("loc", "positional", None),
        ("coeff", "positional", None),
        ("max_iter", "keyword-only", "64"),
        ("tol", "keyword-only", "1e-06"),
        ("step", "keyword-only", "0.1"),
        ("order", "keyword-only", "3"),
        ("bound", "keyword-only", "'dct2'"),
    ],
    "dt_spline_gaussnewton": [
        ("loc", "positional", None),
        ("coeff", "positional", None),
        ("max_iter", "keyword-only", "64"),
        ("tol", "keyword-only", "1e-06"),
        ("order", "keyword-only", "3"),
        ("bound", "keyword-only", "'dct2'"),
    ],
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("op", sorted(DT_SPLINE_SIGNATURES))
def test_dt_spline_signature_is_the_canonical_one(backend, op):
    """``order``, not ``spline``; and the refiner knobs have defaults.

    ``fastfields.cupy`` spelled the spline order ``spline`` (the C++ hub's
    name for it, and the name the raw ``fastfields.dlpack`` binding keeps)
    and made ``max_iter``/``tol``/``step`` required positionals. The wrapper
    layer calls it ``order`` in every other op on every backend --
    ``resample``, ``pull``, ``spline_coeff`` -- so these are ``order`` too.
    """
    actual = SURFACE[backend].get(op)
    assert actual is not None, f"fastfields.{backend} has no {op!r}"
    expected = DT_SPLINE_SIGNATURES[op]
    assert actual == expected, (
        f"fastfields.{backend}.{op}{_render(actual)} does not match the "
        f"canonical {op}{_render(expected)} (API_CONTRACT.md, "
        '"Canonical operations")'
    )


# --------------------------------------------------------------------------- #
# 3. The buffer-naming conventions auto depends on                           #
# --------------------------------------------------------------------------- #
#
# fastfields.auto forwards **kwargs, so the *name* of the first argument is
# part of the contract: `ff.dt_euclidean(inp=a)` must not work on one backend
# and raise TypeError on another. These three conventions are what the
# wrappers settled on; they are checked by name so a drift is caught even
# where the rest of the signature still agrees.

#: ops whose first parameter is the single input array
FIRST_IS_INP = (
    "dt_euclidean",
    "dt_l1",
    "resample",
    "restriction",
    "spline_coeff",
    "pull",
    "push",
    "grad",
) + tuple(
    f"{family}_{sign}{op}{suffix}"
    for family in ("field", "flow")
    for op in ("matvec", "diag", "kernel")
    for sign in ("add", "sub")
    for suffix in ("", "_")
)
#: ops whose first parameter is a buffer both read and written in place
FIRST_IS_INP_OUT = ("dt_euclidean_", "dt_l1_", "spline_coeff_", "sym_solve_")
#: ops whose first parameter is an accumulator written through
FIRST_IS_OUT0 = ("sym_addmatvec_", "sym_submatvec_")

BUFFER_CONVENTIONS = (
    [(op, "inp") for op in FIRST_IS_INP]
    + [(op, "inp_out") for op in FIRST_IS_INP_OUT]
    + [(op, "out0") for op in FIRST_IS_OUT0]
)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "op,expected", BUFFER_CONVENTIONS, ids=[c[0] for c in BUFFER_CONVENTIONS]
)
def test_first_parameter_follows_the_buffer_convention(backend, op, expected):
    signature = SURFACE[backend].get(op)
    assert signature is not None, f"fastfields.{backend} has no {op!r}"
    assert signature[0][0] == expected, (
        f"fastfields.{backend}.{op}'s first parameter is "
        f"{signature[0][0]!r}, expected {expected!r}: fastfields.auto "
        "forwards keyword arguments unchanged, so the first parameter's name "
        "is part of the cross-backend contract"
    )


# --------------------------------------------------------------------------- #
# 4. The presence matrix                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("op", sorted(PARTIAL_OPS))
def test_partial_op_is_on_exactly_the_documented_backends(op):
    actual = frozenset(b for b in BACKENDS if op in SURFACE[b])
    assert actual == PARTIAL_OPS[op], (
        f"{op} is on {sorted(actual)} but PARTIAL_OPS says "
        f"{sorted(PARTIAL_OPS[op])}. If a backend gained it, remove it from "
        "PARTIAL_OPS here and from the 'does not expose' list in "
        "API_CONTRACT.md and the package docstring; if one lost it, that is "
        "a regression."
    )


def test_every_other_op_is_on_all_three_backends():
    """Anything not in ``PARTIAL_OPS`` must be exposed everywhere."""
    gaps = {}
    for op in ALL_OPS:
        if op in PARTIAL_OPS:
            continue
        missing = sorted(b for b in BACKENDS if op not in SURFACE[b])
        if missing:
            gaps[op] = missing
    assert not gaps, (
        "operation(s) missing from some backends and not declared in "
        f"PARTIAL_OPS: {gaps}"
    )


# --------------------------------------------------------------------------- #
# 5. fastfields.auto's routing table matches the backends                    #
# --------------------------------------------------------------------------- #


def test_auto_forwards_every_shared_op():
    missing = sorted(
        op
        for op in ALL_OPS
        if op not in AUTO_TABLES
        and op not in AUTO_EXEMPT
        and all(op in SURFACE[b] for b in BACKENDS)
    )
    assert not missing, (
        f"fastfields.auto has no dispatcher for {missing}, which every "
        "backend exposes. Add one, or add the name to AUTO_EXEMPT with the "
        "reason it cannot be dispatched."
    )


def test_auto_routing_table_names_only_backends_that_have_the_op():
    """A table entry promising an op a backend lacks is a runtime-only error.

    ``_util.make_dispatcher`` resolves ``getattr(backend, attr)`` at call
    time, so a stale entry raises ``NotImplementedError`` from inside a user's
    call rather than failing here. Check it statically instead.
    """
    wrong = []
    for op, table in sorted(AUTO_TABLES.items()):
        for backend, attribute in sorted(table.items()):
            if backend not in BACKENDS:
                continue
            if attribute not in SURFACE[backend]:
                wrong.append(f"auto.{op} -> fastfields.{backend}.{attribute}")
    assert not wrong, (
        f"fastfields.auto routes to operation(s) the backend does not "
        f"export: {wrong}"
    )


def test_auto_routing_table_omits_no_backend_that_has_the_op():
    """The converse: a backend that has an op must be in its table."""
    missing = []
    for op, table in sorted(AUTO_TABLES.items()):
        for backend in BACKENDS:
            if op in SURFACE[backend] and backend not in table:
                missing.append(f"auto.{op} does not route {backend}")
    assert not missing, (
        "fastfields.auto would raise NotImplementedError for a backend that "
        f"does have the operation: {missing}"
    )


# --------------------------------------------------------------------------- #
# 6. Return order, where it can be read statically                           #
# --------------------------------------------------------------------------- #
#
# numpy and cupy return plain local names, so their return *order* is
# comparable from the source. torch returns `_SomeFunction.apply(...)`, whose
# order lives in that Function's forward, so it is out of reach here and is
# covered by tests/test_new_surface_torch.py at runtime (torch does import in
# CI). test_dt_spline_return_order_cupy.py pins cupy against the canonical
# order by value; this pins numpy and cupy against *each other*, for every
# tuple-returning op.


def _returned_tuples(backend: str, op: str) -> "set | None":
    package = _portion(backend)
    _names, origin = _reexported(package / "__init__.py")
    node = _defs(package).get(op)
    if not isinstance(node, ast.FunctionDef):
        return None
    tuples = set()
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Return) or inner.value is None:
            continue
        if not isinstance(inner.value, ast.Tuple):
            continue
        if not all(
            isinstance(element, ast.Name) for element in inner.value.elts
        ):
            return None
        tuples.add(tuple(element.id for element in inner.value.elts))
    return tuples


@pytest.mark.parametrize("op", SHARED_OPS)
def test_numpy_and_cupy_return_the_same_tuple(op):
    numpy_tuples = _returned_tuples("numpy", op)
    cupy_tuples = _returned_tuples("cupy", op)
    if not numpy_tuples or not cupy_tuples:
        return  # no tuple-literal return to compare; nothing to pin here
    assert numpy_tuples == cupy_tuples, (
        f"{op} returns {sorted(cupy_tuples)} on cupy and "
        f"{sorted(numpy_tuples)} on numpy. fastfields.auto returns the "
        "backend's result unchanged, so a different order means a dispatched "
        "call's return value depends on which array type was passed "
        "(fastfields#38)."
    )
