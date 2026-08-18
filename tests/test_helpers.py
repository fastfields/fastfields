"""Tests for the shared enums and argument-normalisation helpers.

Ported from ``fastfields-dlpack``'s ``tests/test_bindings.py``
(``test_shared_helpers``) when this package was split out of it. The four
sections below are the same four the original covered; they are separate test
functions here because this is now the only suite gating these functions,
rather than one case inside a suite that mostly exercises the bindings.

Nothing here imports an array backend or the compiled bindings -- the package
under test is stdlib-only, and this suite keeps it that way.
"""

import pytest

from fastfields.helpers import (
    Bound,
    Spline,
    anchor_scale_shift,
    as_bound,
    as_spline,
    check_ndim,
    infer_ndim,
    normalize_shape,
    resolve_out_spatial,
)


def test_import_is_stdlib_only():
    """Importing the package must not drag in numpy/torch/cupy or dlpack."""
    import sys

    forbidden = ("numpy", "torch", "cupy", "fastfields.dlpack")
    assert not [m for m in forbidden if m in sys.modules]


def test_enums():
    assert int(Spline.Nearest) == 0
    assert int(Spline.Cubic) == 3
    assert int(Spline.SeventhOrder) == 7
    assert int(Bound.Zero) == 0
    assert int(Bound.DCT2) == 3
    assert int(Bound.NoCheck) == 7


def test_order_bound_normalisation():
    """order / bound accept an int, an enum member or a friendly name."""
    assert as_spline("cubic") == 3
    assert as_spline(Spline.Linear) == 1
    assert as_spline(3) == 3
    assert as_bound("dct2") == int(Bound.DCT2)
    assert as_bound("wrap") == int(Bound.DFT)
    assert as_bound(Bound.Replicate) == 1
    # aliases are matched case- and whitespace-insensitively
    assert as_spline("  CUBIC ") == 3
    assert as_bound("Neumann") == int(Bound.DCT2)

    for bad in ("nope", 99, -1):
        with pytest.raises(ValueError):
            as_spline(bad)
        with pytest.raises(ValueError):
            as_bound(bad)


def test_ndim_inference_and_check():
    assert infer_ndim(None, None, [4, 4]) == 2
    assert infer_ndim(None, 2.0, None) == 1
    assert infer_ndim(3, None, None) == 3
    # a scalar shape/factor implies nothing, so it falls back to 1
    assert infer_ndim(None, None, 4) == 1
    assert infer_ndim(None, None, None) == 1

    check_ndim(1, 2)  # in range -> no raise
    check_ndim(2, 2)
    for bad in (0, 3):
        with pytest.raises(ValueError):
            check_ndim(bad, 2)


def test_factor_shape_to_output_shape():
    assert normalize_shape(4, 2) == [4, 4]
    assert normalize_shape([4, 5], 2) == [4, 5]
    with pytest.raises(ValueError):
        normalize_shape([4, 5], 3)

    assert resolve_out_spatial((5,), 1, 2, None) == (10,)
    assert resolve_out_spatial((5, 5), 2, None, 10) == (10, 10)
    assert resolve_out_spatial((7,), 1, None, None) == (7,)  # identity
    # an output size never rounds down to zero
    assert resolve_out_spatial((3,), 1, 0.01, None) == (1,)
    with pytest.raises(ValueError):
        resolve_out_spatial((5, 5), 2, [2.0, 2.0, 2.0], None)


def test_anchor_scale_shift():
    """anchor -> (per-dim scale, scalar shift), matching interpol.resize."""
    assert anchor_scale_shift("centers", (8,), (4,), 1) == ([7 / 3], 0.0)
    assert anchor_scale_shift("edges", (8,), (4,), 1) == ([2.0], 0.5)
    assert anchor_scale_shift("first", (8,), (4,), 1) == ([2.0], 0.0)
    assert anchor_scale_shift("last", (8,), (4,), 1) == ([2.0], 1.0)

    # the abbreviation resolves to the same mapping as the full name
    for name, abbr in [
        ("centers", "c"),
        ("edges", "e"),
        ("first", "f"),
        ("last", "l"),
    ]:
        assert anchor_scale_shift(abbr, (8,), (4,), 1) == anchor_scale_shift(
            name, (8,), (4,), 1
        )

    # "centers" degenerates to scale 1.0 rather than dividing by zero
    assert anchor_scale_shift("centers", (1,), (4,), 1) == ([1.0], 0.0)
    assert anchor_scale_shift("centers", (8,), (1,), 1) == ([1.0], 0.0)

    with pytest.raises(ValueError, match="anchor"):
        anchor_scale_shift("nope", (8,), (4,), 1)
    with pytest.raises(ValueError, match="anchor"):
        anchor_scale_shift("", (8,), (4,), 1)
