# fastfields

The Python side of the fastfields project: one distribution, `fastfields`,
shipping five subpackages of the `fastfields` namespace.

| import | for |
|---|---|
| `fastfields.auto` | dispatches on the type of the array you pass |
| `fastfields.numpy` | NumPy arrays, CPU |
| `fastfields.torch` | PyTorch tensors, with autograd |
| `fastfields.cupy` | CuPy arrays in CUDA device memory |
| `fastfields.helpers` | the shared `Spline`/`Bound` enums and argument normalisers |

One more subpackage, `fastfields.dlpack`, merges into the same namespace from a
**separate** distribution and repository,
[`fastfields-dlpack`](https://github.com/fastfields/fastfields-dlpack): it is
the compiled nanobind bindings over the C++/CUDA library, and it stays separate
because it needs a per-platform wheel matrix. Everything here is pure Python
and ships as one universal `py3-none-any` wheel.

No distribution ships a `fastfields/__init__.py`. Each ships only its
`fastfields/<sub>/` directories, so `fastfields` stays a native (PEP 420)
namespace package and the two installs merge into one importable namespace. See
[`docs/monorepo.md`](docs/monorepo.md) for why that matters and what else the
five-into-one merge did and did not change.

## Install

```bash
pip install "fastfields[numpy]" \
    --extra-index-url https://fastfields.github.io/whl/cpu/
```

`[numpy]`, `[torch]`, `[cupy]` and `[all]` pull the array libraries, which are
optional: `fastfields.auto` dispatches over whichever of them you installed.
`fastfields-dlpack` is the only hard dependency.

The extra selects the array library; the `--extra-index-url` selects which
compiled `fastfields-dlpack` lane you get (`cpu`, or a `cu*` CUDA lane). Those
are two independent choices, and `pip` cannot tie them together — so
`fastfields[cupy]` installs perfectly happily on the CPU lane and then fails at
`import fastfields.cupy`, with a message naming the lane to install instead.
That is explained, with the reasoning, in
[`docs/monorepo.md`](docs/monorepo.md#the-one-wrinkle-an-extra-is-not-a-lane).

## fastfields.auto

`fastfields.auto` exposes one unified function per operation and dispatches on
the **type of the first array argument**:

```python
import numpy as np
import fastfields.auto as ff

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
`fastfields.auto` works with whatever subset of the array libraries is
installed. Every backend exposes the same canonical `dt_*` / `sym_*` names, and
each unified name maps straight onto that function in the selected backend.
Arguments are forwarded unchanged, so each dispatched call shares the signature
of the selected backend's function. A clear `TypeError` is raised for an
unrecognised array type, and a clear `ImportError` if the needed array library
is not installed. `fastfields.auto.dlpack_backend()` reports which compiled
lane you are on.

## Develop

```bash
pip install fastfields-dlpack \
    --extra-index-url https://fastfields.github.io/whl/cpu/
pip install -e ".[test]"
cd /tmp && python -m pytest "$OLDPWD/tests" -q   # matches CI
ruff check . && ruff format --check .
```

The suite is run from a neutral working directory on purpose: that way it
exercises the *installed* package rather than the source tree it happens to sit
next to. The `fastfields.cupy` runtime tests need CuPy and a CUDA device and
skip without them; on a CPU-lane `fastfields-dlpack` the whole
`fastfields.cupy` import is expected to refuse, and the suite asserts that it
refuses *usefully*.

Documentation is [zensical](https://zensical.org); `docs/` builds to
<https://fastfields.github.io/fastfields/>.
