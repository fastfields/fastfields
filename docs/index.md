# fastfields

`fastfields` is the umbrella distribution of the fastfields project. It provides **`fastfields.any`**, a backend-dispatching interface over the per-backend wrapper packages, and anchors the `fastfields` PEP 420 namespace the other distributions merge into. Every function dispatches on the **type of the first array argument** (`numpy.ndarray` -> `fastfields.numpy`, `torch.Tensor` -> `fastfields.torch`, `cupy.ndarray` -> `fastfields.cupy`). Backends are imported lazily, so it works with whatever subset is installed.

## Installation

```bash
pip install fastfields[numpy]      # or [torch], [cupy], [all]
```

## Usage

```python
import numpy as np
import fastfields.any as ff

mat = np.random.randn(5, 6).astype(np.float64)   # packed C=3 symmetric
vec = np.random.randn(5, 3)
out = ff.sym_matvec(mat, vec)      # -> fastfields.numpy.sym_matvec

import torch
t_mat = torch.randn(5, 6, dtype=torch.float64)
t_vec = torch.randn(5, 3, dtype=torch.float64)
out_t = ff.sym_matvec(t_mat, t_vec)   # -> fastfields.torch.sym_matvec
```

See the [API reference](api/index.md) for the full list of operations.
