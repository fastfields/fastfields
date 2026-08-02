# Cross-backend API contract

`fastfields.any` dispatches on the type of the first array argument and forwards
the call **unchanged** to the matching backend (`fastfields.numpy`,
`fastfields.torch`, `fastfields.cupy`). For that to be safe, every backend must
agree on the name, argument order, and meaning of each canonical operation. This
document states that contract; `tests/test_conformance.py` enforces the parts of
it that can be checked automatically.

The golden rule: **a backend may _add_ trailing keyword parameters, but never
rename, drop, or reorder a canonical parameter.** Anything a backend adds must be
optional so an `any` call written against the canonical signature keeps working.

## Canonical operations

Every backend exposes these names with these signatures. `order`/`bound` accept
an `int`, a `Spline`/`Bound` enum, or a friendly string (`"cubic"`, `"dct2"`, …);
`Spline` and `Bound` are re-exported by every backend.

| Operation | Canonical signature | Notes |
| --- | --- | --- |
| `dt_euclidean` | `(x, voxel_spacing=1.0, *, ...)` | Euclidean distance transform along the last axis. |
| `dt_l1` | `(x, voxel_spacing=1.0, *, ...)` | L1 distance transform. |
| `dt_mesh` | `(points, vertices, faces, ...)` | Point-to-triangular-mesh distance. `faces` is normalized to int64. |
| `sym_matvec` | `(mat, vec)` | Compact-symmetric matrix-vector product. |
| `sym_solve` | `(mat, vec, weight=None)` | Solve `(mat + diag(weight)) x = vec`. |
| `sym_invert` | `(mat)` | Invert a compact-symmetric matrix. |
| `resample` | `(x, factor=None, shape=None, *, order=2, bound="dct2", ndim=None, anchor="centers", shift=None, scale=None)` | Spline resample. `factor` **or** `shape` (mutually exclusive); a positional `2` is a **factor** on every backend. |
| `restriction` | same as `resample` | Adjoint of `resample` at reciprocal scale. |
| `spline_coeff` | `(x, order=3, bound="dct2", *, ...)` | Spline-coefficient prefilter along the last axis. |
| `Spline`, `Bound` | enums | Re-exported by every backend. |

Positional call `resample(x, 2)` therefore means *factor 2* on numpy, torch and
cupy alike — the historical footgun (a bare `2` meaning an output shape on
torch/cupy) is gone (see fastfields-lib#17 C2).

## In-place policy (per backend, by design)

In-place (`_`-suffixed) variants mutate the caller's buffer and return it.

### The rule (per-op, not per-backend)

An in-place op is exposed **iff its backward does not require the pre-mutation
value of the tensor being mutated.**

Concretely, for `out <- f(out, ...)` the in-place form is safe under autograd
iff `d f / d out` is a constant that does not depend on `out` — i.e. `f` is
*additive* in `out`:

| shape of `f` | `d f/d out` | pre-mutation value needed? | in-place exposed? |
| --- | --- | --- | --- |
| `out += g(...)` / `out -= g(...)` | `+/- I` | **no** | **yes**, all backends |
| `out *= g(...)`, or any nonlinear `f(out)` | depends on `out` | **yes** | no — out-of-place only |

This is exactly why PyTorch itself ships `Tensor.add_` as a fully autograd-safe
in-place op while e.g. an in-place nonlinearity has to stash its input.

**The previous blanket rule — "torch omits every in-place op, because in-place
mutation does not compose with autograd" — was wrong** and has been replaced by
the per-op test above. Additive accumulation composes with autograd perfectly
well.

### Consequences for torch

- `{field,flow}_{add,sub}{matvec,diag}_` **are** exposed on torch. They are
  additive in the mutated tensor, so their `torch.autograd.Function` saves
  nothing for backward (no `save_for_backward` at all — that absence *is* the
  safety argument) and returns the incoming gradient unchanged for the
  accumulated-into tensor.
- Any in-place op implemented as a `torch.autograd.Function` **must** call
  `ctx.mark_dirty()` on the tensor it mutates, so the version counter is bumped
  and a stale save elsewhere raises instead of silently producing a wrong
  gradient.
- Torch's ordinary **leaf rule** is unchanged and is *not* a fastfields policy:
  a leaf tensor with `requires_grad=True` cannot be mutated in place, exactly as
  for `x.add_(y)`. Use the out-of-place spelling in that case.
- Ops whose backward *does* need the original values — `sym_solve_`,
  `sym_invert_`, `spline_coeff_`, `dt_euclidean_` — remain **out-of-place only**
  on torch. That is the rule biting, not an arbitrary exclusion.

### One kernel, two spellings

For the regulariser accumulate ops there is a **single** C primitive, and it is
in-place only (`ff::{field,flow}_{add,sub}{matvec,diag}_`, mirroring the
original jitfields `op='+'`/`op='-'` entry points). The two Python spellings
differ only in whether the caller's tensor is passed straight through or cloned
first:

```python
def field_addmatvec_(inp, field, ...):   # in-place
    return _prim(inp, field, ...)

def field_addmatvec(inp, field, ...):    # out-of-place
    return _prim(inp.clone(), field, ...)
```

There is deliberately no separate "return a fresh tensor" kernel below Python.
This holds identically on numpy, torch and cupy.

### Availability

`{field,flow}_{add,sub}{matvec,diag}` and their `_`-suffixed in-place forms are
available on **numpy, torch and cupy alike**. Other `_`-suffixed ops remain
backend-specific (see below).

Because these live outside the canonical set, `fastfields.any` never routes to a
`_`-suffixed op implicitly; they are backend-specific extensions.

## Backend-specific extensions (allowed)

Beyond the canonical set, backends may add:

- **numpy** — `sym_matvec_backward`, `sym_channels_from_packed`,
  `dt_spline_{table,brent,gaussnewton}`.
- **torch** — the regulariser accumulate set
  (`{field,flow}_{add,sub}{matvec,diag}` and their `_` in-place forms),
  which are autograd-safe by the rule above. Nothing else beyond the
  canonical set.
- **cupy** — the in-place set above plus `current_stream_ptr` and the
  `dt_spline_*` variants.

These are not part of the `any` contract; code that relies on them targets a
specific backend on purpose.

## What the conformance test checks

`tests/test_conformance.py`:

1. **Surface** — every installed backend exposes each canonical name (and the
   `Spline`/`Bound` enums).
2. **In-place policy** — numpy exposes `sym_addmatvec_`/`sym_submatvec_`; torch
   exposes neither (the documented autograd exclusion).
3. **Equivalence** — for the shared ops, a numpy call and the equivalent torch
   call (dispatched through `fastfields.any`) produce numerically identical
   results, so `any` means the same thing on both. cupy is skipped where no GPU
   is available.
