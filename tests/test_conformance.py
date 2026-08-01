"""Cross-backend API conformance tests (fastfields#4).

Enforces the contract in ``API_CONTRACT.md``: every backend exposes the same
canonical operation names, the in-place policy differs as documented
(numpy/cupy have ``_``-suffixed ops, torch does not), and a canonical call
dispatched through ``fastfields.any`` gives the same result on numpy and torch.

numpy is a hard test dependency; torch is imported via ``importorskip`` so the
suite still runs where torch is absent; cupy is skipped (needs a CUDA GPU).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import fastfields.any as ff

# The canonical operations every backend must expose, plus the re-exported
# enums. A backend may ADD trailing optional params or extra ops, never drop or
# rename these.
CANONICAL = [
    "dt_euclidean",
    "dt_l1",
    "dt_mesh",
    "sym_matvec",
    "sym_solve",
    "sym_invert",
    "resample",
    "restriction",
    "spline_coeff",
    "Spline",
    "Bound",
]

BACKENDS = ["numpy", "torch", "cupy"]


def _import_backend(name: str):
    """Import ``fastfields.<name>`` or skip if the backend isn't installed."""
    try:
        return importlib.import_module(f"fastfields.{name}")
    except Exception:  # ImportError, or cupy raising with no GPU
        pytest.skip(f"backend fastfields.{name} not available")


# --------------------------------------------------------------------------- #
# 1. Canonical surface                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", CANONICAL)
def test_backend_exposes_canonical_op(backend, name):
    mod = _import_backend(backend)
    assert hasattr(mod, name), (
        f"fastfields.{backend} is missing canonical name {name!r}"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_enums_reexported(backend):
    mod = _import_backend(backend)
    assert int(mod.Spline.Cubic) == 3
    assert int(mod.Bound.DCT2) == 3


# --------------------------------------------------------------------------- #
# 2. In-place policy                                                          #
# --------------------------------------------------------------------------- #


def test_numpy_exposes_inplace_sym():
    mod = _import_backend("numpy")
    assert hasattr(mod, "sym_addmatvec_")
    assert hasattr(mod, "sym_submatvec_")


# The regulariser accumulate ops are additive in the tensor they mutate, so
# their backward never needs the pre-mutation value -- they are autograd-safe
# in place and are exposed on every backend, torch included.
# See API_CONTRACT.md, "In-place policy".
_TORCH_INPLACE_ALLOWED = frozenset(
    f"{fam}_{sign}{op}_"
    for fam in ("field", "flow")
    for op in ("matvec", "diag")
    for sign in ("add", "sub")
)


def test_torch_inplace_ops_are_only_the_autograd_safe_ones():
    # torch must not grow in-place ops whose backward would need the
    # pre-mutation value of the mutated tensor (e.g. sym_solve_, sym_invert_,
    # spline_coeff_): those stay out-of-place only (API_CONTRACT.md).
    mod = _import_backend("torch")
    exposed = {
        name
        for name in dir(mod)
        if name.endswith("_") and not name.startswith("_")
    }
    unexpected = exposed - _TORCH_INPLACE_ALLOWED
    assert not unexpected, (
        f"torch exposes in-place op(s) that are not known to be "
        f"autograd-safe: {sorted(unexpected)!r}"
    )


def test_torch_exposes_the_autograd_safe_inplace_ops():
    # The converse: the accumulate set must actually be present on torch.
    mod = _import_backend("torch")
    missing = _TORCH_INPLACE_ALLOWED - set(dir(mod))
    assert not missing, f"torch is missing in-place op(s): {sorted(missing)!r}"


# --------------------------------------------------------------------------- #
# 3. numpy vs torch numerical equivalence (dispatched through fastfields.any) #
# --------------------------------------------------------------------------- #


def _packed_spd(batch, c, seed):
    """Return a (batch, packed) stack of well-conditioned SPD matrices."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((batch, c, c))
    m = a @ np.transpose(a, (0, 2, 1)) + c * np.eye(c)  # SPD, well-conditioned
    packed = np.empty((batch, c * (c + 1) // 2), dtype=np.float64)
    for b in range(batch):
        idx = 0
        for k in range(c):
            packed[b, idx] = m[b, k, k]
            idx += 1
        for i in range(c):
            for j in range(i + 1, c):
                packed[b, idx] = m[b, i, j]
                idx += 1
    return packed


# Each entry: (id, builds a tuple of numpy positional args, kwargs, call name).
def _cases():
    dt_line = np.array([[0.0, 1e30, 1e30, 0.0, 1e30, 0.0]], dtype=np.float64)
    ramp = np.arange(8.0)
    packed = _packed_spd(5, 3, seed=0)
    vec = np.random.default_rng(1).standard_normal((5, 3))
    return [
        ("dt_euclidean", (dt_line,), {}),
        ("dt_l1", (dt_line,), {}),
        ("sym_matvec", (packed, vec), {}),
        ("sym_solve", (packed, vec), {}),
        ("sym_invert", (packed,), {}),
        ("resample", (ramp,), dict(factor=2, order="linear")),
        ("restriction", (ramp,), dict(factor=2, order="linear")),
        ("spline_coeff", (ramp,), dict(order="cubic")),
    ]


@pytest.mark.parametrize(
    "op,args,kwargs", _cases(), ids=[c[0] for c in _cases()]
)
def test_numpy_torch_equivalence(op, args, kwargs):
    torch = pytest.importorskip("torch")
    _import_backend("torch")  # ensure fastfields.torch is importable

    np_out = getattr(ff, op)(*args, **kwargs)

    targs = tuple(torch.as_tensor(a, dtype=torch.float64) for a in args)
    t_out = getattr(ff, op)(*targs, **kwargs)

    assert isinstance(np_out, np.ndarray)
    assert torch.is_tensor(t_out)
    np.testing.assert_allclose(
        np_out, t_out.detach().cpu().numpy(), rtol=1e-6, atol=1e-6
    )


# --------------------------------------------------------------------------- #
# pushpull + regulariser dispatch (added when those op families were exposed) #
# --------------------------------------------------------------------------- #


def _pull_case():
    inp = np.array([[0.0], [10.0], [20.0], [30.0]])
    grid = np.array([[0.5], [1.5], [2.5]])
    return inp, grid


def test_any_exposes_pushpull_and_reg():
    for name in [
        "pull",
        "push",
        "count",
        "grad",
        "field_matvec",
        "flow_matvec",
    ]:
        assert hasattr(ff, name), name


def test_any_pull_numpy_torch_equivalence():
    torch = pytest.importorskip("torch")
    _import_backend("torch")
    inp, grid = _pull_case()
    on = ff.pull(inp, grid, order=1)
    ot = ff.pull(torch.as_tensor(inp), torch.as_tensor(grid), order=1)
    np.testing.assert_allclose(
        on, ot.detach().cpu().numpy(), rtol=1e-6, atol=1e-6
    )


def test_any_field_matvec_numpy_torch_equivalence():
    torch = pytest.importorskip("torch")
    _import_backend("torch")
    f = np.random.default_rng(0).standard_normal((8, 2))
    on = ff.field_matvec(f, absolute=[2.0, 3.0], ndim=1)
    ot = ff.field_matvec(torch.as_tensor(f), absolute=[2.0, 3.0], ndim=1)
    np.testing.assert_allclose(
        on, ot.detach().cpu().numpy(), rtol=1e-6, atol=1e-6
    )
