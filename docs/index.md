# fastfields

**fastfields** is the Python interface to the fastfields toolkit. In the
common case you import one module, `fastfields.auto`, and call it on whatever
array you have: it looks at the array's type, NumPy or PyTorch, and runs the
operation on the matching backend. Write your code once and it works across
frameworks.

One `pip install fastfields` gives you all of it: `fastfields.auto`, the
per-framework modules `fastfields.numpy`, `fastfields.torch` and
`fastfields.cupy`, and the shared `fastfields.helpers`. Only the backends you
install are *loaded*, so `fastfields[numpy]` alone is a perfectly good setup.
Today the tested, working path is the **CPU** one, with NumPy and PyTorch. The
CuPy and GPU path is implemented but not yet published as a wheel or validated
on hardware; see [Status](#status) below.

This is the Python section of the fastfields documentation. The [project
overview](https://fastfields.github.io/) has the per-operation status matrix
and the roadmap. The layers underneath are documented separately: the [raw
DLPack binding](https://fastfields.github.io/dlpack/) and the [C++/CUDA
library](https://fastfields.github.io/lib/).

## Install

```sh
# CPU (NumPy)
pip install "fastfields[numpy]" \
    --extra-index-url https://fastfields.github.io/whl/cpu/
```

Use `fastfields[torch]` for PyTorch, or `fastfields[all]` for all three.

The extra picks the array library, and the `--extra-index-url` picks the
compiled `fastfields-dlpack` build. These are two separate choices, and the
second one is the one that matters for GPU work; see [the packaging
note](monorepo.md#the-one-wrinkle-an-extra-is-not-a-lane). A GPU wheel lane is
planned but **not published yet**; see [Status](#status).

## A first example

```python
import numpy as np
from fastfields import auto as ff

feature = np.full((256, 256), np.inf, dtype="float32")
feature[:, 128] = 0.0  # features are 0, everything else is +inf

dist = ff.dt_euclidean(feature)  # dispatched to the NumPy backend
```

The same call on a `torch.Tensor`, for instance
`ff.dt_euclidean(torch.as_tensor(feature))`, is dispatched to PyTorch.

## What's inside

| Operation | Functions |
|---|---|
| **Distance transforms** | `dt_euclidean` and `dt_l1`; point-to-spline `dt_spline_table`, `dt_spline_brent` and `dt_spline_gaussnewton`; point-to-mesh `dt_mesh` |
| **Positive-definite linear algebra** | `sym_matvec`, `sym_matvec_backward`, `sym_solve` and `sym_invert`, over whole fields of small symmetric matrices |
| **Resampling** | `resample` (spline up- and down-sampling), `restriction` (its adjoint) and `spline_coeff` (the coefficient prefilter) |
| **Spline sampling (pushpull)** | `pull` and `push` (gather and its adjoint scatter), `count` and `grad`: the building blocks of image warping |
| **Regularisers** | `field_matvec` and `flow_matvec` (absolute, membrane and bending energies on multi-channel fields and on vector flows) and the operators built on them: diagonals, stencils, accumulate forms, the weighted `*_rls` variants and the regularised system `(H + L) x = g` |

Each call forwards straight to the selected backend, so it shares that
backend's signature and behaviour: autograd on PyTorch, CUDA streams on CuPy.
In-place spellings are dispatched too (`dt_euclidean_`, `sym_solve_`,
`spline_coeff_`, the `*_add…_` and `*_sub…_` accumulate forms). They update
their first argument, which is also the array the backend is chosen from.

If you only use one framework, import its module directly:
[`fastfields.numpy`](numpy.md), [`fastfields.torch`](torch.md) or
[`fastfields.cupy`](cupy.md). They are in the same install. What dispatch
leaves out is described on the [`fastfields.auto`](auto.md#what-is-not-dispatched)
page.

!!! note "`dt_spline_*` returns `(dist, time)` on every backend"
    The three point-to-spline operators return a 2-tuple, and all three
    namespaces return it in that order. `fastfields.cupy` returned
    `(time, dist)` until [fastfields#38][ff38]. A dispatcher that forwards
    arguments unchanged has nowhere to normalise a *result*, so the meaning of
    a dispatched return value depended on which array type you passed. The
    names are part of the canonical surface that the cross-backend
    conformance suite checks, so a namespace cannot quietly drift again.

    Their *signatures* agreed later than their return order did.
    `fastfields.cupy` spelled the spline order `spline` and made
    `max_iter`, `tol` and `step` required positional arguments, where numpy and
    torch had keyword-only `order` and `bound` with defaults. All three now
    use the numpy and torch spelling, and any drift in parameter names, kinds,
    order or defaults fails CI in `tests/test_signature_conformance.py`
    instead of at a user's call.

    [ff38]: https://github.com/fastfields/fastfields/issues/38

## Status

fastfields is **alpha**. The table below is what is actually exposed and tested
today, mirroring the candour of the internal migration matrix — so you know
before you install.

| Operation | NumPy (CPU) | PyTorch (CPU) | CuPy (GPU) |
|---|:---:|:---:|:---:|
| Distance — Euclidean / L1 | ✅ | ✅ | 🧭 |
| Distance — point-to-mesh / point-to-spline | 🧪 | 🧪 | 🧭 |
| Positive-definite linear algebra (`sym_*`) | ✅ | ✅ | 🧭 |
| Resampling (`resample` / `restriction` / `spline_coeff`) | ✅ | ✅ | 🧭 |
| Pushpull (spline gather / scatter / grad) | ✅ | ✅ | 🧭 |
| Regularisers (membrane / bending energies) | ✅ | ✅ | 🧭 |

**✅ works, covered by the CPU test suite** &nbsp;·&nbsp; **🧪 exposed but not
yet covered by tests** (shape contracts still being firmed up — use with care)
&nbsp;·&nbsp; **🧭 planned** (see below).

What "planned" means here:

- **GPU / CuPy** (the only remaining 🧭). The CUDA kernels and the CuPy wrapper
  exist and compile + link — every op family, pushpull and the regularisers
  included, is exposed on all three backends — but there is no GPU in CI, so no
  GPU wheel is published yet and nothing has run on real hardware. The wheel
  lanes are deferred until that changes.

Everything marked ✅ is exercised by a brute-force reference test on CPU.

See the [API reference](api/index.md) for full signatures and options, and
[one distribution, five namespaces](monorepo.md) for how the packaging fits
together.
