# fastfields.helpers

`fastfields.helpers` is the part of the fastfields stack that does not depend
on an array library. It holds the `Spline` and `Bound` enums and the helpers
that turn user-friendly arguments into what the compiled library expects. It is
shared by `fastfields.numpy`, `fastfields.torch`, `fastfields.cupy` and the
`fastfields.auto` dispatcher.

## Install

```sh
pip install fastfields
```

It ships with the `fastfields` distribution, so there is nothing extra to
install. The module imports only the standard library: no array backend and
none of the compiled bindings.

## A first example

```python
from fastfields.helpers import Bound, Spline, as_bound, as_spline

as_spline("cubic")  # 3
as_bound("wrap")  # 6, which is int(Bound.DFT)
as_spline(Spline.Linear)  # 1
```

The normalisers are what let every `order` and `bound` argument in the other
namespaces accept an `int`, an enum member or a name.

## What's inside

| Group | Names |
|---|---|
| **Enums** | [`Spline`][fastfields.helpers.Spline] (orders 0 to 7) and [`Bound`][fastfields.helpers.Bound] (seven boundary conditions) |
| **Normalisers** | [`as_spline`][fastfields.helpers.as_spline] and [`as_bound`][fastfields.helpers.as_bound] accept an `int`, an enum member or a name |
| **Shapes** | [`normalize_shape`][fastfields.helpers.normalize_shape], [`infer_ndim`][fastfields.helpers.infer_ndim], [`check_ndim`][fastfields.helpers.check_ndim] and [`resolve_out_spatial`][fastfields.helpers.resolve_out_spatial] work out the arguments of a resampling call |
| **Anchors** | [`anchor_scale_shift`][fastfields.helpers.anchor_scale_shift] turns an `interpol.resize` anchor into a per-axis `scale` and a scalar `shift` |
| **Compute backend** | [`is_cuda_backend`][fastfields.helpers.is_cuda_backend] and [`cuda_backend_message`][fastfields.helpers.cuda_backend_message] decide whether the installed `fastfields-dlpack` is a CUDA build, and word the error when it is not |

## Why it is a separate namespace

These helpers used to live inside `fastfields-dlpack`, next to the compiled
extension. A one-line fix to a string alias or a rounding rule could then only
reach users through a full compiled-wheel release on every platform. They were
split out to escape that, and the escape survives the merge into one
distribution: `fastfields` is still a single universal `py3-none-any` wheel
with no compiled content, so fixing a helper is an ordinary pure-Python
release.

For the same reason the module must keep depending on nothing. It imports no
array library and, in particular, never `fastfields.dlpack`; a subprocess test
enforces that.

See the [API reference](api/helpers.md) for the full signatures.
