"""Tests for fastfields.any dispatch.

Verify that dispatching a numpy array and a torch tensor through
``fastfields.any`` matches the direct backend calls, and that unknown types /
missing backends give clear errors.
"""

import numpy as np
import pytest

import fastfields.any as ff
import fastfields.numpy as ffn


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
    # unified name dt_euclidean -> numpy euclidean_distance_transform
    x = np.array([[0, np.inf, np.inf, 0, np.inf]], dtype=np.float64)
    out_any = ff.dt_euclidean(x, voxel_spacing=1.0)
    out_direct = ffn.euclidean_distance_transform(x, voxel_spacing=1.0)
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


def test_unknown_type_raises():
    with pytest.raises(TypeError):
        ff.sym_matvec([1.0, 2.0, 3.0], [1.0, 2.0])


def test_enums_reexported():
    assert int(ff.Spline.Cubic) == 3
    assert int(ff.Bound.DCT2) == 3
    assert ff.__version__
