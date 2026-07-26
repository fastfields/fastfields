# fastfields

**fastfields** is the one-import way to use the toolkit: `fastfields.any` looks
at whatever array you hand it — NumPy or PyTorch — and runs the operation on the
matching backend. Write your code once and it works across frameworks.

Only the backends you install are loaded, so `fastfields[numpy]` alone is a
perfectly good setup. Today the tested, working path is **CPU** (NumPy and
PyTorch); the CuPy/GPU path is implemented but not yet published as a wheel or
validated on hardware — see [Status](#status) below.

## Install

```sh
# CPU (NumPy)
pip install "fastfields[numpy]" \
    --extra-index-url https://fastfields.github.io/whl/cpu/
```

Use `fastfields[torch]` for PyTorch. A GPU (CuPy / CUDA) wheel lane is planned
but **not published yet** — see [Status](#status).

## Use it

```python
import numpy as np
from fastfields import any as ff

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

Each call forwards straight to the selected backend, so it shares that backend's
signature and behaviour (autograd on PyTorch, CUDA streams on CuPy). Prefer a
single framework? Use `fastfields.numpy`, `fastfields.torch` or `fastfields.cupy`
directly.

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
| Pushpull (spline gather / scatter / grad) | 🧭 | 🧭 | 🧭 |
| Regularisers (membrane / bending energies) | 🧭 | 🧭 | 🧭 |

**✅ works, covered by the CPU test suite** &nbsp;·&nbsp; **🧪 exposed but not
yet covered by tests** (shape contracts still being firmed up — use with care)
&nbsp;·&nbsp; **🧭 planned** (see below).

¹ PyTorch exposes point-to-mesh (`dt_mesh`) but not the point-to-spline
variants.

What "planned" means here:

- **GPU / CuPy.** The CUDA kernels and the CuPy wrapper exist and compile + link,
  but there is no GPU in CI, so no GPU wheel is published yet and nothing has run
  on real hardware. The wheel lanes are deferred until that changes.
- **Pushpull and regularisers.** These are implemented in the C++ library but are
  not yet bound in the Python bindings, so no backend exposes them today.

Everything marked ✅ is exercised by a brute-force reference test on CPU.

See the [API reference](api/index.md) for full signatures and options.
