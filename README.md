# fastfields

`fastfields` is the umbrella distribution of the fastfields project. It provides
**`fastfields.any`**, a backend-dispatching interface over the per-backend
wrapper packages, and anchors the `fastfields` PEP 420 namespace that the other
distributions merge into:

| import              | distribution         | backend        |
|---------------------|----------------------|----------------|
| `fastfields.dlpack` | `fastfields-dlpack`  | raw bindings   |
| `fastfields.numpy`  | `fastfields-numpy`   | numpy          |
| `fastfields.torch`  | `fastfields-torch`   | torch          |
| `fastfields.cupy`   | `fastfields-cupy`    | cupy           |
| `fastfields.any`    | `fastfields`         | dispatch       |

No distribution ships a `fastfields/__init__.py`; each ships only its
`fastfields/<sub>/` subpackage, so `fastfields` stays a native (PEP 420)
namespace package and the five installs merge into one importable namespace.

## fastfields.any

`fastfields.any` exposes one unified function per operation and dispatches on the
**type of the first array argument**:

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

* `numpy.ndarray` -> `fastfields.numpy`
* `torch.Tensor`  -> `fastfields.torch`
* `cupy.ndarray`  -> `fastfields.cupy`

Backends are imported **lazily** (only the one whose array type is passed), so
`fastfields.any` works with whatever subset of the backends is installed. Only
`fastfields-dlpack` is a hard dependency; pull the array backends via extras:

```bash
pip install fastfields[numpy]      # or [torch], [cupy], [all]
```

Every backend exposes the same canonical `dt_*` / `sym_*` names, and each
unified name (e.g. `dt_euclidean`, `dt_mesh`) maps straight onto that function
in the selected backend. Arguments are forwarded unchanged, so each dispatched
call shares the signature of the selected backend's function. A clear
`TypeError` is raised for an unrecognised array type, and a clear `ImportError`
if the needed backend is not installed.
