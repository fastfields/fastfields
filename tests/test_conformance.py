"""Cross-backend API conformance tests (fastfields#4).

Enforces the contract in ``API_CONTRACT.md``: every backend exposes the same
canonical operation names; numpy, torch and cupy now expose the *same*
``dt_euclidean_``/``dt_l1_``/``spline_coeff_``/``sym_solve_``/``sym_invert_``
in-place surface (full parity -- no backend silently omits a non-
differentiable op); torch's non-differentiable ops (``dt_euclidean``,
``dt_l1``, ``dt_mesh``, ``sym_invert``, and their in-place forms) raise a
clear ``RuntimeError`` from ``.backward()`` rather than being omitted or
silently wrong; and a canonical call dispatched through ``fastfields.auto``
gives the same result on numpy and torch.

numpy is a hard test dependency; torch is imported via ``importorskip`` so the
suite still runs where torch is absent; cupy is skipped (needs a CUDA GPU).
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

import fastfields.auto as ff

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


@pytest.mark.parametrize("backend", BACKENDS)
def test_dt_mesh_flags_are_not_keyword_only(backend):
    # signed/naive/return_nearest must be positional-or-keyword on every
    # backend (API_CONTRACT.md, "Canonical operations"): a positional call
    # must not work on some backends and raise TypeError on others. numpy
    # used to make these keyword-only; fixed in fastfields-numpy#24.
    mod = _import_backend(backend)
    sig = inspect.signature(mod.dt_mesh)
    for name in ("signed", "naive", "return_nearest"):
        assert sig.parameters[name].kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ), f"fastfields.{backend}.dt_mesh's {name!r} must not be keyword-only"


# --------------------------------------------------------------------------- #
# 2. In-place policy                                                          #
# --------------------------------------------------------------------------- #


# numpy and cupy have no autograd, so both expose the same in-place surface:
# the regulariser accumulate set (field/flow, shared with torch -- see below)
# plus the posdef accumulate ops (sym_addmatvec_/sym_submatvec_, numpy/cupy
# only -- torch has no sym_addmatvec/sym_submatvec at all) plus
# dt_euclidean_/dt_l1_/spline_coeff_/sym_solve_/sym_invert_. Prior to
# fastfields-numpy#24 numpy spelled the first three as an `inplace=` keyword
# and was missing the last two outright -- an accidental gap, not a
# deliberate difference (API_CONTRACT.md, "Availability"). This set is
# enforced identically for numpy and cupy so it can't silently drift apart.
_NUMPY_CUPY_INPLACE = frozenset(
    {
        "sym_addmatvec_",
        "sym_submatvec_",
        "sym_solve_",
        "sym_invert_",
        "dt_euclidean_",
        "dt_l1_",
        "spline_coeff_",
    }
)

# The non-differentiable in-place ops that fastfields#4 asked to be exposed
# on torch too, for parity with numpy/cupy -- previously omitted "for
# autograd reasons" even though there was no gradient to protect (see
# API_CONTRACT.md, "Non-differentiable ops: exposed everywhere, backward
# raises"). sym_addmatvec_/sym_submatvec_ are deliberately excluded here:
# torch has no sym_addmatvec/sym_submatvec at all (a separate, open gap --
# see API_CONTRACT.md, "Availability" -- not something fastfields#4 covers).
_TORCH_NONDIFF_INPLACE = _NUMPY_CUPY_INPLACE - {
    "sym_addmatvec_",
    "sym_submatvec_",
}

# The regulariser accumulate ops are additive in the tensor they mutate, so
# their backward never needs the pre-mutation value -- they are autograd-safe
# in place and are exposed on every backend, torch included.
# See API_CONTRACT.md, "In-place policy".
_TORCH_ACCUMULATE_INPLACE = frozenset(
    f"{fam}_{sign}{op}_"
    for fam in ("field", "flow")
    for op in ("matvec", "diag")
    for sign in ("add", "sub")
)

_TORCH_INPLACE_ALLOWED = _TORCH_ACCUMULATE_INPLACE | _TORCH_NONDIFF_INPLACE

# Ops with no supported gradient (on any backend): forward always runs, but
# calling `.backward()` through the output must raise a clear RuntimeError
# naming the op -- never a silent wrong answer, never a generic/cryptic
# autograd internal error, and never simply omitted. See API_CONTRACT.md,
# "Non-differentiable ops: exposed everywhere, backward raises".
_TORCH_NONDIFF_OUT_OF_PLACE = frozenset(
    {"dt_euclidean", "dt_l1", "sym_invert"}
)


def test_numpy_exposes_inplace_sym():
    mod = _import_backend("numpy")
    assert hasattr(mod, "sym_addmatvec_")
    assert hasattr(mod, "sym_submatvec_")


def test_numpy_exposes_the_full_nonautograd_inplace_set():
    mod = _import_backend("numpy")
    missing = _NUMPY_CUPY_INPLACE - set(dir(mod))
    assert not missing, f"numpy is missing in-place op(s): {sorted(missing)!r}"


def test_cupy_exposes_the_full_nonautograd_inplace_set():
    mod = _import_backend("cupy")
    missing = _NUMPY_CUPY_INPLACE - set(dir(mod))
    assert not missing, f"cupy is missing in-place op(s): {sorted(missing)!r}"


def test_torch_inplace_ops_are_only_the_known_safe_ones():
    # torch must not grow undocumented in-place ops: only the regulariser
    # accumulate set (autograd-safe by construction) and the non-
    # differentiable set (dt_euclidean_/dt_l1_/spline_coeff_/sym_solve_/
    # sym_invert_ -- exposed for parity, guarded by a backward that either
    # works (spline_coeff_/sym_solve_) or raises (the rest) -- see
    # API_CONTRACT.md, "Consequences for torch" / "Non-differentiable ops").
    mod = _import_backend("torch")
    exposed = {
        name
        for name in dir(mod)
        if name.endswith("_") and not name.startswith("_")
    }
    unexpected = exposed - _TORCH_INPLACE_ALLOWED
    assert not unexpected, (
        f"torch exposes in-place op(s) that are not in the documented "
        f"allowed set: {sorted(unexpected)!r}"
    )


def test_torch_exposes_the_autograd_safe_inplace_ops():
    # The accumulate set must actually be present on torch.
    mod = _import_backend("torch")
    missing = _TORCH_ACCUMULATE_INPLACE - set(dir(mod))
    assert not missing, f"torch is missing in-place op(s): {sorted(missing)!r}"


def test_torch_matches_numpy_cupy_nondiff_inplace_surface():
    # fastfields#4: torch must no longer silently omit the non-differentiable
    # in-place ops that numpy/cupy expose.
    mod = _import_backend("torch")
    missing = _TORCH_NONDIFF_INPLACE - set(dir(mod))
    assert not missing, (
        f"torch is missing non-differentiable in-place op(s) that numpy/"
        f"cupy expose: {sorted(missing)!r}"
    )


@pytest.mark.parametrize("name", sorted(_TORCH_NONDIFF_OUT_OF_PLACE))
def test_torch_nondiff_op_backward_raises(name):
    torch = pytest.importorskip("torch")
    mod = _import_backend("torch")
    # A length-3 float64 vector is a valid input to all three: dt_euclidean/
    # dt_l1 only care about dtype, sym_invert needs a valid packed matrix
    # (len 3 == C=2).
    x = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64, requires_grad=True)
    out = getattr(mod, name)(x)
    assert out.requires_grad, f"{name} should still build a graph on forward"
    with pytest.raises(RuntimeError, match=name):
        out.sum().backward()


# Of the non-differentiable in-place set, only these three actually raise on
# backward: spline_coeff_/sym_solve_ are differentiable in place (covered by
# test_torch_sym_solve_inplace_and_spline_coeff_inplace_are_differentiable).
_TORCH_NONDIFF_INPLACE_RAISES = _TORCH_NONDIFF_INPLACE - {
    "spline_coeff_",
    "sym_solve_",
}


@pytest.mark.parametrize("name", sorted(_TORCH_NONDIFF_INPLACE_RAISES))
def test_torch_nondiff_inplace_op_backward_raises(name):
    torch = pytest.importorskip("torch")
    mod = _import_backend("torch")
    base = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    x = base + torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)  # non-leaf
    out = getattr(mod, name)(x)
    assert out is x
    with pytest.raises(RuntimeError, match=name):
        out.sum().backward()


def test_torch_sym_solve_inplace_and_spline_coeff_inplace_are_differentiable():
    # The two non-differentiable-in-place-set members that ARE actually
    # differentiable in place: verify gradcheck passes (not just "doesn't
    # raise") -- see API_CONTRACT.md, "Consequences for torch".
    torch = pytest.importorskip("torch")
    mod = _import_backend("torch")

    packed = torch.tensor([[2.0, 3.0, 0.5]], dtype=torch.float64)
    vec = torch.randn(1, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda v: mod.sym_solve_(v.clone(), packed), (vec,)
    )

    x = torch.randn(1, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda t: mod.spline_coeff_(t.clone(), 3, "dct2"), (x,)
    )


# ------------------------------------------------------------------------- #
# 3. numpy vs torch numerical equivalence (dispatched via fastfields.auto)  #
# ------------------------------------------------------------------------- #


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


def test_auto_exposes_pushpull_and_reg():
    for name in [
        "pull",
        "push",
        "count",
        "grad",
        "field_matvec",
        "field_relax",
        "field_matvec_rls",
        "field_diag_rls",
        "field_relax_rls",
        "flow_matvec",
        "flow_relax",
        "flow_matvec_rls",
        "flow_diag_rls",
        "flow_relax_rls",
    ]:
        assert hasattr(ff, name), name


def test_auto_pull_numpy_torch_equivalence():
    torch = pytest.importorskip("torch")
    _import_backend("torch")
    inp, grid = _pull_case()
    on = ff.pull(inp, grid, order=1)
    ot = ff.pull(torch.as_tensor(inp), torch.as_tensor(grid), order=1)
    np.testing.assert_allclose(
        on, ot.detach().cpu().numpy(), rtol=1e-6, atol=1e-6
    )


def test_auto_field_matvec_numpy_torch_equivalence():
    torch = pytest.importorskip("torch")
    _import_backend("torch")
    f = np.random.default_rng(0).standard_normal((8, 2))
    on = ff.field_matvec(f, absolute=[2.0, 3.0], ndim=1)
    ot = ff.field_matvec(torch.as_tensor(f), absolute=[2.0, 3.0], ndim=1)
    np.testing.assert_allclose(
        on, ot.detach().cpu().numpy(), rtol=1e-6, atol=1e-6
    )


def test_auto_field_matvec_rls_numpy_torch_equivalence():
    # field_matvec_rls dispatches on `inp` (its first array arg), exactly
    # like field_matvec; a real (non-uniform) weight map must reach the
    # same result on both backends.
    torch = pytest.importorskip("torch")
    _import_backend("torch")
    rng = np.random.default_rng(12)
    H, W, C = 6, 7, 2
    f = rng.standard_normal((H, W, C))
    w = 0.5 + np.abs(rng.standard_normal((H, W, 1)))
    kw = dict(absolute=[0.3, 0.4], membrane=[1.0, 0.7], ndim=2)
    on = ff.field_matvec_rls(f, w, **kw)
    ot = ff.field_matvec_rls(torch.as_tensor(f), torch.as_tensor(w), **kw)
    np.testing.assert_allclose(
        on, ot.detach().cpu().numpy(), rtol=1e-6, atol=1e-6
    )


def test_auto_field_relax_numpy_torch_equivalence():
    # field_relax is dispatched on its first array arg and mutates it in
    # place; both backends must land on the same refined field.
    torch = pytest.importorskip("torch")
    _import_backend("torch")
    rng = np.random.default_rng(11)
    H, W, C, hdiag = 5, 6, 2, 6.0
    hes = np.zeros((H, W, C * (C + 1) // 2))
    hes[..., 0] = hdiag
    hes[..., 1] = hdiag
    grd = rng.standard_normal((H, W, C))
    kw = dict(absolute=[0.3, 0.4], membrane=[0.7, 0.5], ndim=2, nb_iter=20)
    on = ff.field_relax(np.zeros((H, W, C)), hes, grd, **kw)
    ot = ff.field_relax(
        torch.zeros(H, W, C, dtype=torch.float64),
        torch.as_tensor(hes),
        torch.as_tensor(grd),
        **kw,
    )
    np.testing.assert_allclose(
        on, ot.detach().cpu().numpy(), rtol=1e-6, atol=1e-6
    )


def test_auto_flow_matvec_rls_numpy_torch_equivalence():
    # The flow-side RLS/JRLS ops dispatch on their first array arg just like
    # the field-side ones. `wgt` is always joint here (trailing size-1 axis).
    torch = pytest.importorskip("torch")
    _import_backend("torch")
    rng = np.random.default_rng(13)
    H, W, D = 6, 7, 2
    f = rng.standard_normal((H, W, D))
    w = 0.5 + np.abs(rng.standard_normal((H, W, 1)))
    kw = dict(absolute=0.3, membrane=1.0, shears=0.5, div=0.4, ndim=2)
    on = ff.flow_matvec_rls(f, w, **kw)
    ot = ff.flow_matvec_rls(torch.as_tensor(f), torch.as_tensor(w), **kw)
    np.testing.assert_allclose(
        on, ot.detach().cpu().numpy(), rtol=1e-6, atol=1e-6
    )


def test_auto_flow_diag_rls_numpy_torch_equivalence():
    # flow_diag_rls is dispatchable (unlike plain flow_diag, which builds
    # from a shape): its first argument is the weight *array*.
    torch = pytest.importorskip("torch")
    _import_backend("torch")
    rng = np.random.default_rng(14)
    H, W = 6, 7
    w = 0.5 + np.abs(rng.standard_normal((H, W, 1)))
    kw = dict(absolute=0.3, membrane=1.0, shears=0.5, div=0.4, ndim=2)
    on = ff.flow_diag_rls(w, **kw)
    ot = ff.flow_diag_rls(torch.as_tensor(w), **kw)
    assert on.shape == (H, W, 2)
    np.testing.assert_allclose(
        on, ot.detach().cpu().numpy(), rtol=1e-6, atol=1e-6
    )


def test_auto_flow_relax_rls_numpy_torch_equivalence():
    torch = pytest.importorskip("torch")
    _import_backend("torch")
    rng = np.random.default_rng(15)
    H, W, D, hdiag = 5, 6, 2, 8.0
    hes = np.zeros((H, W, D * (D + 1) // 2))
    hes[..., 0] = hdiag
    hes[..., 1] = hdiag
    grd = rng.standard_normal((H, W, D))
    w = 0.5 + np.abs(rng.standard_normal((H, W, 1)))
    kw = dict(
        absolute=0.3, membrane=0.7, shears=1.0, div=0.5, ndim=2, nb_iter=20
    )
    on = ff.flow_relax_rls(np.zeros((H, W, D)), hes, grd, w, **kw)
    ot = ff.flow_relax_rls(
        torch.zeros(H, W, D, dtype=torch.float64),
        torch.as_tensor(hes),
        torch.as_tensor(grd),
        torch.as_tensor(w),
        **kw,
    )
    np.testing.assert_allclose(
        on, ot.detach().cpu().numpy(), rtol=1e-6, atol=1e-6
    )
