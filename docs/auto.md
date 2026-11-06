# fastfields.auto

**`fastfields.auto`** is the backend-dispatching interface: every function looks
at the **type of the first array argument** and forwards the call, unchanged, to
the matching per-framework package. Write the call once and it runs on whichever
array library the caller happens to be using.

| you pass | it calls |
|---|---|
| `numpy.ndarray` | [`fastfields.numpy`](numpy.md) |
| `torch.Tensor` | [`fastfields.torch`](torch.md) |
| `cupy.ndarray` | [`fastfields.cupy`](cupy.md) |

## Install

```sh
pip install "fastfields[numpy]" \
    --extra-index-url https://fastfields.github.io/whl/cpu/
```

`fastfields.auto` ships with the `fastfields` distribution and needs no extra of
its own — but it has nothing to dispatch *to* until at least one backend is
installed, so pick the extra for the array library you use (`[numpy]`,
`[torch]`, `[cupy]`, or `[all]`). It dispatches over exactly the subset you
installed.

## Use it

```python
import numpy as np
from fastfields import auto as ff

mask = np.zeros((256, 256), "float32")
mask[:, 128] = 1.0

dist = ff.dt_euclidean(mask)      # -> runs on fastfields.numpy
```

Hand the same function a `torch.Tensor` and it runs on `fastfields.torch`
instead, with autograd, and returns a tensor. Arguments are forwarded
unchanged, so each call has the signature of the backend function it selects —
spline order and boundary arguments accept an `int`, a `Spline` / `Bound` enum,
or a string like `"cubic"` or `"dct2"`, exactly as they do on a concrete
backend.

## When to use it, and when to use a backend directly

Reach for `fastfields.auto` when the array type is **not yours to choose**:
library code that must accept whatever its caller hands it, a function that runs
over both NumPy and PyTorch data, or a script you want to move between CPU and
GPU without rewriting imports.

Use [`fastfields.numpy`](numpy.md), [`fastfields.torch`](torch.md) or
[`fastfields.cupy`](cupy.md) directly when you already know the array type. You
get a typed API, the backend's own documented signatures, and an import error at
import time rather than a dispatch error on first call. Nothing is lost by doing
so — `fastfields.auto` adds dispatch, not capability, and every operation it
reaches is an operation one of those three already has.

## Backends are imported lazily

Detection is by the array type's **top-level module name**, so importing
`fastfields.auto` does not import NumPy, PyTorch or CuPy — only the backend for
an array you actually pass gets imported, and only once. That is what makes it
usable with any subset of the backends installed.

The failures are correspondingly specific:

- an object that is not a NumPy, PyTorch or CuPy array raises `TypeError`
  naming the type it was given;
- an array whose backend package (or that package's own dependency) is not
  installed raises `ImportError` naming what to install;
- an operation a backend does not have raises `NotImplementedError` naming the
  backends that do have it — never a bare `AttributeError` from a module the
  caller never mentioned.

## What it does not dispatch

Two kinds of operation cannot be reached through `fastfields.auto`, both
deliberately:

- **Builders that take a shape, not an array.** `field_diag`, `flow_diag`,
  `field_kernel` and `flow_kernel` allocate their output from a rank and a set
  of penalty weights, so there is no array to dispatch on. Call them on a
  concrete backend and pass the result into the dispatched operators. Their
  `*_add…` / `*_sub…` accumulate forms *do* take an array and are dispatched.
- **Adjoints that exist for autograd.** `pull_backward`, `push_backward`,
  `count_backward` and `grad_backward` are wired into
  `fastfields.torch`'s `autograd.Function` rather than exported, since only
  PyTorch has an autograd engine. The supported spelling is
  `fastfields.auto.pull(...)` followed by `.backward()` on a torch tensor.

A few operations exist on some backends only and are dispatched with a partial
table, raising `NotImplementedError` that names the backends which have them:
`field_cg` (numpy), `field_precond_` / `flow_precond_` and `sym_matvec_backward`
(numpy and cupy). Listing them with a clear failure keeps the asymmetry visible.

## Which lane am I on?

`fastfields.auto` never requires a particular compute backend — unlike
`fastfields.cupy`, which refuses to import on a CPU-only build, and
`fastfields.torch`, which refuses a CUDA tensor on one. It only reports:

```python
from fastfields import auto as ff

ff.dlpack_backend()      # "cpu", "cu126", ... or None
```

`None` means the installed `fastfields-dlpack` build carries no lane label, or
that it cannot be imported at all. Only the **CPU** lane is published today, so
`fastfields.cupy` and CUDA `torch` tensors are not usable yet — see the
[project status matrix](https://fastfields.github.io/#status).

See the [API reference](api/auto.md) for the full list of dispatched operations
and their signatures.
