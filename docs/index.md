# fastfields

**fastfields** is the one-import way to use the toolkit: `fastfields.auto` looks
at whatever array you hand it — NumPy or PyTorch — and runs the operation on the
matching backend. Write your code once and it works across frameworks.

One `pip install fastfields` gives you all of it: `fastfields.auto` plus the
per-framework modules `fastfields.numpy`, `fastfields.torch` and
`fastfields.cupy`, and the shared `fastfields.helpers`. Only the backends you
install are *loaded*, so `fastfields[numpy]` alone is a perfectly good setup.
Today the tested, working path is **CPU** (NumPy and PyTorch); the CuPy/GPU
path is implemented but not yet published as a wheel or validated on hardware
— see [Status](#status) below.

This is the Python section of the fastfields documentation. The
[project overview](https://fastfields.github.io/) has the per-operation status
matrix and the roadmap; the layers underneath are documented separately, as the
[raw DLPack binding](https://fastfields.github.io/dlpack/) and the
[C++/CUDA library](https://fastfields.github.io/lib/).

## Install

```sh
# CPU (NumPy)
pip install "fastfields[numpy]" \
    --extra-index-url https://fastfields.github.io/whl/cpu/
```

Use `fastfields[torch]` for PyTorch, or `fastfields[all]` for all three.

The extra picks the array library; the `--extra-index-url` picks the compiled
`fastfields-dlpack` lane. Those are two separate choices, and the second one
is the one that matters for GPU work — see [the monorepo
note](monorepo.md#the-one-wrinkle-an-extra-is-not-a-lane). A GPU (CuPy / CUDA)
wheel lane is planned but **not published yet** — see [Status](#status).

## Use it

```python
import numpy as np
from fastfields import auto as ff

mask = np.zeros((256, 256), "float32")
mask[:, 128] = 1.0

dist = ff.dt_euclidean(mask)      # dispatched to the NumPy backend

import torch
tmask = torch.zeros(256, 256)
tmask[:, 128] = 1.0
tdist = ff.dt_euclidean(tmask)    # same call, dispatched to PyTorch
```

## What's inside

| Operation | Functions |
|---|---|
| **Distance transforms** | `dt_euclidean`, `dt_l1`; point-to-spline `dt_spline_table` / `dt_spline_brent` / `dt_spline_gaussnewton`; point-to-mesh `dt_mesh` |
| **Positive-definite linear algebra** | `sym_matvec`, `sym_matvec_backward`, `sym_solve`, `sym_invert` over whole fields of small symmetric matrices |
| **Resampling** | `resample` (spline up/down-sampling), `restriction` (its adjoint), `spline_coeff` (coefficient prefilter) |
| **Pushpull** | `pull` / `push` (spline gather & its adjoint scatter), `count`, `grad` — the building blocks of image warping/sampling |
| **Regularisers** | `field_matvec` / `flow_matvec` (absolute / membrane / bending energies on fields & vector flows) and their `*_diag` preconditioners |

Each call forwards straight to the selected backend, so it shares that backend's
signature and behaviour (autograd on PyTorch, CUDA streams on CuPy). Prefer a
single framework? Use [`fastfields.numpy`](numpy.md),
[`fastfields.torch`](torch.md) or [`fastfields.cupy`](cupy.md) directly — they
are in the same install.

In-place spellings are dispatched too (`dt_euclidean_`, `sym_solve_`,
`spline_coeff_`, the `*_add…_`/`*_sub…_` accumulate forms): they mutate their
first argument, which is also the array the backend is chosen from.

### What is not dispatched here, and why

`fastfields.auto` picks a backend from the type of the first array argument
and forwards everything unchanged. Two kinds of operation cannot be reached
through it, and both are deliberate.

**Builders that take a shape, not an array** — `field_diag`, `flow_diag`,
`field_kernel`, `flow_kernel` allocate their output from a rank and a set of
penalty weights, so there is no array to dispatch on. Call them on a concrete
backend and pass the result into the dispatched operators. Their
`*_add…`/`*_sub…` accumulate forms *do* take an array and are dispatched.

**Adjoints that exist for autograd** — `pull_backward`, `push_backward`,
`count_backward` and `grad_backward` are the gradients of the pushpull
operators. Only `fastfields.torch` has an autograd engine, and there they are
wired into the forward op's `autograd.Function` rather than exported, so
there is nothing uniform to dispatch. Call `fastfields.auto.pull(...)` on a
torch tensor and `.backward()`, or reach for `fastfields.dlpack` raw.

Operations only *some* backends have are still dispatched, with a partial
table that raises `NotImplementedError` naming the backends that do have
them: `field_cg` (CPU-only in the C++ library, so numpy), `field_precond_` /
`flow_precond_` (numpy and cupy), `sym_matvec_backward` (numpy and cupy).
Listing them with a clear failure beats omitting them, which would make the
asymmetry invisible from here.

And if a backend turns out not to have an operation this module says it does
— a mixed install, where `fastfields.auto` is newer than the wrapper — the
call raises `NotImplementedError` saying which package to upgrade, not a bare
`AttributeError` from a module you never named.

!!! note "`dt_spline_*` returns `(dist, time)` on every backend"
    The three point-to-spline operators return a 2-tuple, and all three
    namespaces return it in that order. `fastfields.cupy` returned
    `(time, dist)` until [fastfields#38][ff38]: a dispatcher that forwards
    arguments unchanged has nowhere to normalise a *result*, so the meaning of
    a dispatched return value depended on which array type you passed. The
    three names are part of the canonical surface the cross-backend
    conformance suite checks, so a namespace cannot quietly drift again.

    Their *signatures* agreed later than their return order did.
    `fastfields.cupy` spelled the spline order `spline` and made
    `max_iter`/`tol`/`step` required positional arguments, where numpy and
    torch had keyword-only `order`/`bound` with defaults. All three now use
    the numpy/torch spelling, and signature drift — parameter names, kinds,
    order, defaults — fails CI in `tests/test_signature_conformance.py`
    rather than at a user's call.

    [ff38]: https://github.com/fastfields/fastfields/issues/38

## Status

fastfields is **alpha**. The table below is what is actually exposed and tested
today, mirroring the candour of the internal migration matrix — so you know
before you install.

| Operation | NumPy (CPU) | PyTorch (CPU) | CuPy (GPU) |
|---|:---:|:---:|:---:|
| Distance — Euclidean / L1 | ✅ | ✅ | 🧭 |
| Distance — point-to-mesh / point-to-spline | 🧪 | 🧪¹ | 🧭 |
| Positive-definite linear algebra (`sym_*`) | ✅ | ✅ | 🧭 |
| Resampling (`resample` / `restriction` / `spline_coeff`) | ✅ | ✅ | 🧭 |
| Pushpull (spline gather / scatter / grad) | ✅ | ✅ | 🧭 |
| Regularisers (membrane / bending energies) | ✅ | ✅ | 🧭 |

**✅ works, covered by the CPU test suite** &nbsp;·&nbsp; **🧪 exposed but not
yet covered by tests** (shape contracts still being firmed up — use with care)
&nbsp;·&nbsp; **🧭 planned** (see below).

¹ PyTorch exposes point-to-mesh (`dt_mesh`) but not the point-to-spline
variants.

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
