# fastfields-helpers

The shared, **dependency-free** Python layer of the
[fastfields](https://github.com/fastfields) stack: the `Spline` and `Bound`
enums, and the argument-normalisation helpers that every wrapper
(`fastfields.numpy`, `fastfields.torch`, `fastfields.cupy`) and the
`fastfields.auto` dispatcher share.

```python
from fastfields.helpers import Spline, Bound, as_spline, as_bound

as_spline("cubic")        # 3
as_bound("wrap")          # 6  == int(Bound.DFT)
as_spline(Spline.Linear)  # 1
```

## Install

```sh
pip install fastfields-helpers
```

There is nothing to compile and nothing else to install: this package imports
only the standard library.

## Why it exists

These helpers used to live inside `fastfields-dlpack`, alongside the compiled
nanobind extension. That meant a one-line fix to a string alias or a rounding
rule could only reach users through a full compiled-wheel release cycle, built
and published for every platform.

Splitting them out makes that cost disappear: `fastfields-helpers` is a plain
universal wheel, so pure-Python fixes ship on their own. `fastfields-helpers`
and `fastfields-dlpack` are independent siblings — neither depends on the
other.

## Public API

Enums:

- `Spline` — interpolation order, `Nearest` (0) through `SeventhOrder` (7).
- `Bound` — boundary condition: `Zero`, `Replicate`, `DCT1`, `DCT2`, `DST1`,
  `DST2`, `DFT`, `NoCheck`.

Normalisers — each accepts an `int`, an enum member, or a friendly string
alias (`"cubic"`, `"dct2"`, `"wrap"`, `"neumann"`, …) and returns an `int`:

- `as_spline(value)`
- `as_bound(value)`

Shape / dimension resolution:

- `normalize_shape(shape, ndim)` — scalar-or-sequence → `list[int]` of length
  `ndim`.
- `infer_ndim(ndim, factor, shape)` — infer the number of trailing spatial
  dimensions.
- `check_ndim(ndim, arr_ndim)` — validate `1 <= ndim <= arr_ndim`.
- `resolve_out_spatial(spatial_in, ndim, factor, shape)` — output spatial shape
  from a `factor` or an explicit `shape` (mutually exclusive; neither means
  identity).

torch-interpol anchor conventions:

- `anchor_scale_shift(anchor, inshape, outshape, ndim)` — map a
  `centers`/`edges`/`first`/`last` anchor (or its first-letter abbreviation)
  onto the per-dim `scale` and scalar `shift` the resize binding consumes.

See the [API reference](https://fastfields.github.io/fastfields-helpers/api/)
for full signatures.
