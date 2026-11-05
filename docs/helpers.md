# fastfields.helpers

**`fastfields.helpers`** is the shared, dependency-free Python layer of the
fastfields stack: the `Spline` and `Bound` enums, and the argument-normalisation
helpers used by `fastfields.numpy`, `fastfields.torch`, `fastfields.cupy` and
the `fastfields.auto` dispatcher.

## Install

```sh
pip install fastfields
```

It ships with the `fastfields` distribution; there is nothing extra
to install for it. The module itself imports only the standard
library -- no array backend, and none of the compiled bindings.

## Use it

```python
from fastfields.helpers import Spline, Bound, as_spline, as_bound

as_spline("cubic")        # 3
as_bound("wrap")          # 6  == int(Bound.DFT)
as_spline(Spline.Linear)  # 1
```

## Why it exists

These helpers used to live inside `fastfields-dlpack`, next to the
compiled nanobind extension, so a one-line fix to a string alias or a
rounding rule could only reach users through a full compiled-wheel
release cycle on every platform. They were split out to escape that,
and the escape survives the merge: they now ship in the `fastfields`
distribution, which is still a single universal (`py3-none-any`)
wheel with no compiled content, so a fix to them is still an ordinary
pure-Python release.

What this module may depend on has not changed either: nothing. It
imports no array library and, in particular, never
`fastfields.dlpack` -- a subprocess test enforces that.

## What's inside

| Group | Names |
|---|---|
| **Enums** | `Spline` (orders 0–7), `Bound` (8 boundary conditions) |
| **Normalisers** | `as_spline`, `as_bound` — accept an `int`, an enum member or a friendly string |
| **Shape / ndim** | `normalize_shape`, `infer_ndim`, `check_ndim`, `resolve_out_spatial` |
| **Anchors** | `anchor_scale_shift` — torch-interpol anchor → per-dim `scale` + scalar `shift` |

See the [API reference](api/helpers.md) for full signatures and options.
