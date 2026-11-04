# fastfields-helpers

**fastfields-helpers** is the shared, dependency-free Python layer of the
fastfields stack: the `Spline` and `Bound` enums, and the argument-normalisation
helpers used by `fastfields.numpy`, `fastfields.torch`, `fastfields.cupy` and
the `fastfields.auto` dispatcher.

## Install

```sh
pip install fastfields-helpers
```

Nothing to compile, and no other package to install — this one imports only the
standard library.

## Use it

```python
from fastfields.helpers import Spline, Bound, as_spline, as_bound

as_spline("cubic")        # 3
as_bound("wrap")          # 6  == int(Bound.DFT)
as_spline(Spline.Linear)  # 1
```

## Why it exists

These helpers used to live inside `fastfields-dlpack`, next to the compiled
nanobind extension, so a one-line fix to a string alias or a rounding rule could
only reach users through a full compiled-wheel release cycle on every platform.
As a standalone universal wheel they ship on their own instead.

`fastfields-helpers` and `fastfields-dlpack` are independent siblings — neither
depends on the other.

## What's inside

| Group | Names |
|---|---|
| **Enums** | `Spline` (orders 0–7), `Bound` (8 boundary conditions) |
| **Normalisers** | `as_spline`, `as_bound` — accept an `int`, an enum member or a friendly string |
| **Shape / ndim** | `normalize_shape`, `infer_ndim`, `check_ndim`, `resolve_out_spatial` |
| **Anchors** | `anchor_scale_shift` — torch-interpol anchor → per-dim `scale` + scalar `shift` |

See the [API reference](api/index.md) for full signatures and options.
