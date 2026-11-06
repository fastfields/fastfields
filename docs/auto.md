# fastfields.auto

`fastfields.auto` dispatches on the array type. Each function looks at the
type of its first array argument and forwards the call, unchanged, to the
matching per-framework package. Write the call once and it runs on whichever
array library the caller happens to use.

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

`fastfields.auto` ships with the `fastfields` distribution and has no extra of
its own. It has nothing to dispatch *to* until at least one backend is
installed, so pick the extra for the array library you use (`[numpy]`,
`[torch]`, `[cupy]` or `[all]`). It then dispatches over exactly the subset you
installed.

## A first example

```python
import numpy as np
from fastfields import auto as ff

feature = np.full((256, 256), np.inf, dtype="float32")
feature[:, 128] = 0.0  # features are 0, everything else is +inf

dist = ff.dt_euclidean(feature)  # runs on fastfields.numpy
```

Hand the same function a `torch.Tensor` and it runs on `fastfields.torch`
instead, with autograd, and returns a tensor. Arguments are forwarded
unchanged, so each call has the signature of the backend function it selects.
Spline orders and boundary conditions accept an `int`, a `Spline` or `Bound`
member, or a name such as `"cubic"` or `"dct2"`, exactly as they do on a
concrete backend.

## When to use it

Reach for `fastfields.auto` when the array type is not yours to choose: library
code that must accept whatever its caller passes, a function that runs over
both NumPy and PyTorch data, or a script you want to move between CPU and GPU
without rewriting imports.

When you already know the array type, import the backend directly:
[`fastfields.numpy`](numpy.md), [`fastfields.torch`](torch.md) or
[`fastfields.cupy`](cupy.md). You get the backend's own documented signatures,
and a problem with the installation shows up when you import it, not at the
first call. Nothing is lost by doing so. `fastfields.auto` adds dispatch, not
capability: every operation it reaches is one of those three already has.

## Backends are imported lazily

The backend is chosen from the top-level module name of the array's type, so
importing `fastfields.auto` does not import NumPy, PyTorch or CuPy. Only the
backend of an array you actually pass is imported, and only once. That is what
makes the package usable with any subset of them installed.

The failures are specific:

- an object that is not a NumPy, PyTorch or CuPy array raises `TypeError`,
  naming the type it was given;
- an array whose backend package, or that package's own dependency, is not
  installed raises `ImportError`, naming what to install;
- an operation that a backend does not have raises `NotImplementedError`,
  naming the backends that do, and never a bare `AttributeError` from a module
  the caller did not mention.

## What is not dispatched

Two kinds of operation cannot be reached through `fastfields.auto`, and both
are deliberate.

- **Builders that take a shape, not an array.** `field_diag`, `flow_diag`,
  `field_kernel` and `flow_kernel` allocate their output from a rank or shape
  and a set of penalty weights, so there is no array to dispatch on. Call them
  on a concrete backend and pass the result to the dispatched operators. Their
  `*_add…` and `*_sub…` accumulate forms do take an array and are dispatched.
- **Adjoints that exist for autograd.** `pull_backward`, `push_backward`,
  `count_backward` and `grad_backward` are wired into the `autograd.Function`
  of `fastfields.torch` instead of being exported, since only PyTorch has an
  autograd engine. The supported spelling is `fastfields.auto.pull(...)`
  followed by `.backward()` on a torch tensor.

A few operations exist on some backends only. They are dispatched with a
partial table and raise `NotImplementedError` naming the backends that have
them: `field_cg` (numpy only), `field_precond_` and `flow_precond_` (numpy and
cupy), and `sym_matvec_backward` (numpy and cupy). Listing them with a clear
failure keeps the asymmetry visible.

## Which lane am I on?

`fastfields.auto` never requires a particular compute backend. By contrast,
`fastfields.cupy` refuses to import on a CPU-only build, and `fastfields.torch`
refuses a CUDA tensor on one. `fastfields.auto` only reports:

```python
from fastfields import auto as ff

ff.dlpack_backend()  # "cpu", "cu126", ... or None
```

`None` means that the installed `fastfields-dlpack` carries no lane label, or
that it cannot be imported at all. Only the CPU lane is published today, so
`fastfields.cupy` and CUDA `torch` tensors are not usable yet. See the
[project status matrix](https://fastfields.github.io/#status).

The [API reference](api/auto.md) lists every dispatched operation and the
backend page that documents it.
