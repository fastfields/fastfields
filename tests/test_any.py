"""Tests for fastfields.any dispatch.

Verify that dispatching a numpy array and a torch tensor through
``fastfields.any`` matches the direct backend calls, and that unknown types /
missing backends give clear errors.
"""

import fastfields.numpy as ffn
import numpy as np
import pytest

import fastfields.any as ff


def _pack_symmetric(mats):
    B, C, _ = mats.shape
    packed = np.zeros((B, C * (C + 1) // 2), dtype=mats.dtype)
    for b in range(B):
        idx = 0
        for k in range(C):
            packed[b, idx] = mats[b, k, k]
            idx += 1
        for i in range(C):
            for j in range(i + 1, C):
                packed[b, idx] = mats[b, i, j]
                idx += 1
    return packed


def _random_packed(B, C, seed):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((B, C, C))
    m = m + np.transpose(m, (0, 2, 1))
    return _pack_symmetric(m)


def test_dispatch_numpy_matches_backend():
    packed = _random_packed(5, 3, seed=0)
    vec = np.random.default_rng(1).standard_normal((5, 3))

    out_any = ff.sym_matvec(packed, vec)
    out_direct = ffn.sym_matvec(packed, vec)
    assert isinstance(out_any, np.ndarray)
    np.testing.assert_array_equal(out_any, out_direct)


def test_dispatch_numpy_dt_euclidean_name_mapping():
    # unified name dt_euclidean -> numpy dt_euclidean
    x = np.array([[0, np.inf, np.inf, 0, np.inf]], dtype=np.float64)
    out_any = ff.dt_euclidean(x, voxel_spacing=1.0)
    out_direct = ffn.dt_euclidean(x, voxel_spacing=1.0)
    np.testing.assert_array_equal(out_any, out_direct)


def test_dispatch_torch_matches_backend():
    torch = pytest.importorskip("torch")
    import fastfields.torch as fft

    mat = torch.randn(5, 6, dtype=torch.float64)
    vec = torch.randn(5, 3, dtype=torch.float64)

    out_any = ff.sym_matvec(mat, vec)
    out_direct = fft.sym_matvec(mat, vec)
    assert torch.is_tensor(out_any)
    assert torch.equal(out_any, out_direct)


def test_dispatch_selects_by_first_argument():
    # numpy vs torch chosen purely by the first array's type
    torch = pytest.importorskip("torch")

    xn = np.array([[0.0, 1e30, 1e30, 0.0]], dtype=np.float64)
    xt = torch.tensor([[0.0, 1e30, 1e30, 0.0]], dtype=torch.float64)

    on = ff.dt_l1(xn)
    ot = ff.dt_l1(xt)
    assert isinstance(on, np.ndarray)
    assert torch.is_tensor(ot)
    np.testing.assert_allclose(on, ot.numpy(), rtol=1e-6, atol=1e-6)


def test_dispatch_flow_forward_matches_backend():
    # flow_forward/flow_precond dispatch on `mat` (first array arg).
    mat = _random_packed(30, 2, seed=7).reshape(5, 6, 3)
    vec = np.random.default_rng(8).standard_normal((5, 6, 2))
    kw = dict(absolute=0.3, membrane=0.7, shears=1.0, div=0.5, ndim=2)
    np.testing.assert_array_equal(
        ff.flow_forward(mat, vec, **kw), ffn.flow_forward(mat, vec, **kw)
    )


def test_dispatch_flow_precond_matches_backend():
    m = np.random.default_rng(9).standard_normal((30, 2, 2))
    m = np.einsum("bij,bkj->bik", m, m) + 3.0 * np.eye(2)  # SPD
    mat = _pack_symmetric(m).reshape(5, 6, 3)
    vec = np.random.default_rng(10).standard_normal((5, 6, 2))
    kw = dict(absolute=0.3, membrane=0.7, shears=1.0, div=0.5, ndim=2)
    np.testing.assert_array_equal(
        ff.flow_precond(mat, vec, **kw), ffn.flow_precond(mat, vec, **kw)
    )


def test_dispatch_field_precond_forward_matches_backend():
    # field_precond/forward dispatch on `mat`; accumulate on `inp`.
    m = np.random.default_rng(14).standard_normal((30, 2, 2))
    m = np.einsum("bij,bkj->bik", m, m) + 3.0 * np.eye(2)  # SPD
    mat = _pack_symmetric(m).reshape(5, 6, 3)
    vec = np.random.default_rng(15).standard_normal((5, 6, 2))
    kw = dict(absolute=[0.3, 0.4], membrane=[0.7, 0.5], ndim=2)
    np.testing.assert_array_equal(
        ff.field_forward(mat, vec, **kw), ffn.field_forward(mat, vec, **kw)
    )
    np.testing.assert_array_equal(
        ff.field_precond(mat, vec, **kw), ffn.field_precond(mat, vec, **kw)
    )
    np.testing.assert_array_equal(
        ff.field_matvec_add(vec, vec, **kw),
        ffn.field_matvec_add(vec, vec, **kw),
    )


def test_dispatch_flow_accumulate_matches_backend():
    # The _add/_sub/in-place variants dispatch on `inp` (first array arg).
    rng = np.random.default_rng(13)
    flow = rng.standard_normal((5, 6, 2))
    base = rng.standard_normal((5, 6, 2))
    kw = dict(absolute=0.3, membrane=0.7, shears=1.0, div=0.5, ndim=2)
    for name in ("flow_matvec_add", "flow_matvec_sub"):
        np.testing.assert_array_equal(
            getattr(ff, name)(base, flow, **kw),
            getattr(ffn, name)(base, flow, **kw),
        )
    for name in ("flow_diag_add", "flow_diag_sub"):
        np.testing.assert_array_equal(
            getattr(ff, name)(base, **kw), getattr(ffn, name)(base, **kw)
        )
    # in-place through any mutates the passed array and returns it
    a = base.copy()
    assert ff.flow_matvec_add_(a, flow, **kw) is a
    np.testing.assert_array_equal(a, base + ffn.flow_matvec(flow, **kw))


def test_resample_unified_signature_matches_backend():
    # C2: numpy/torch/cupy share the factor/shape/order signature, so
    # any.resample forwards it unchanged and matches the direct backend call.
    x = np.arange(8.0)
    out_any = ff.resample(x, factor=2, order="linear", anchor="centers")
    out_direct = ffn.resample(x, factor=2, order="linear", anchor="centers")
    assert out_any.shape == (16,)
    np.testing.assert_array_equal(out_any, out_direct)
    # `shape=` and a string `order`/`bound` also cross unchanged
    out_shape = ff.resample(x, shape=4, order="cubic", bound="dct2")
    np.testing.assert_array_equal(
        out_shape, ffn.resample(x, shape=4, order="cubic", bound="dct2")
    )


def test_resample_factor_means_factor_on_every_backend():
    # The C2 footgun is gone: a positional 2 is a *factor* on numpy AND torch
    # (previously it meant an output shape on torch/cupy).
    torch = pytest.importorskip("torch")

    xn = np.arange(8.0)
    xt = torch.arange(8.0, dtype=torch.float64)
    on = ff.resample(xn, 2, order="linear")
    ot = ff.resample(xt, 2, order="linear")
    assert on.shape == (16,) and tuple(ot.shape) == (16,)
    np.testing.assert_allclose(on, ot.numpy(), rtol=1e-6, atol=1e-6)


def test_unknown_type_raises():
    with pytest.raises(TypeError):
        ff.sym_matvec([1.0, 2.0, 3.0], [1.0, 2.0])


def test_enums_reexported():
    assert int(ff.Spline.Cubic) == 3
    assert int(ff.Bound.DCT2) == 3
    assert ff.__version__
