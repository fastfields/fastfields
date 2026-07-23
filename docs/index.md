# fastfields

**fastfields** is the one-import way to use the toolkit: `fastfields.any` looks
at whatever array you hand it — NumPy, PyTorch or CuPy — and runs the operation
on the matching backend. Write your code once and it works across frameworks,
CPU and GPU.

Only the backends you install are loaded, so `fastfields[numpy]` alone is a
perfectly good CPU-only setup.

## Install

```sh
# CPU only
pip install "fastfields[numpy]" \
    --extra-index-url https://fastfields.github.io/whl/cpu/

# GPU (CUDA 12.8)
pip install "fastfields[cupy]" \
    --extra-index-url https://fastfields.github.io/whl/cu128/
```

Use `fastfields[torch]` for PyTorch, or `fastfields[all]` for everything.

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

See the [API reference](api/index.md) for full signatures and options.
