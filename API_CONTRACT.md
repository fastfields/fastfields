# Cross-backend API contract

`fastfields.auto` dispatches on the type of the first array argument and forwards
the call **unchanged** to the matching backend (`fastfields.numpy`,
`fastfields.torch`, `fastfields.cupy`). For that to be safe, every backend must
agree on the name, argument order, and meaning of each canonical operation. This
document states that contract; `tests/test_conformance.py` enforces the parts of
it that can be checked automatically.

The golden rule: **a backend may _add_ trailing keyword parameters, but never
rename, drop, or reorder a canonical parameter.** Anything a backend adds must be
optional so an `auto` call written against the canonical signature keeps working.

## Canonical operations

Every backend exposes these names with these signatures. `order`/`bound` accept
an `int`, a `Spline`/`Bound` enum, or a friendly string (`"cubic"`, `"dct2"`, …);
`Spline` and `Bound` are re-exported by every backend.

| Operation | Canonical signature | Notes |
| --- | --- | --- |
| `dt_euclidean` | `(x, voxel_spacing=1.0, *, ...)` | Euclidean distance transform along the last axis. |
| `dt_l1` | `(x, voxel_spacing=1.0, *, ...)` | L1 distance transform. |
| `dt_mesh` | `(points, vertices, faces, signed=True, naive=False, return_nearest=False, ...)` | Point-to-triangular-mesh distance. `faces` is normalized to int64. `signed`/`naive`/`return_nearest` are positional-or-keyword on every backend (not keyword-only) — see fastfields#4. |
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

Unless a parameter is explicitly marked keyword-only in the table above (the
`*` in `resample`'s signature), it is **positional-or-keyword on every
backend**. `dt_mesh`'s `signed`/`naive`/`return_nearest` used to be
keyword-only on numpy only, which meant a positional call worked on
torch/cupy and raised `TypeError` on numpy — an accidental divergence with no
technical justification, fixed in `fastfields-numpy` PR #24 (fastfields#4).

## Why the raw `fastfields.dlpack` binding signature differs from the wrappers

The three friendly wrappers agree with each other on argument order (`resample
(inp, factor, shape, ...)`, `inp` first) but *not* with the raw
`fastfields.dlpack` binding underneath them, whose signature is
`resample(out, inp, spline, bound, shift, scale, ndim, stream=0)` — the output
buffer first. **This is the one divergence in the stack that is genuinely
load-bearing, not an oversight:**

- The binding layer never allocates memory itself. It cannot: numpy, torch and
  cupy each have their own allocator (torch's caching allocator, cupy's memory
  pool, ...), and only the framework-specific wrapper knows which one to use
  and on which device. So the binding *must* receive a pre-allocated `out`
  buffer from its caller — there is no way to make it allocate output the way
  a pythonic `resample(inp, ...) -> out` signature implies. Putting the
  output-shaped, write-only argument first mirrors the project's own
  `DLTensor&` C++ convention throughout the stack (`ff::<fn>(out, in, ...)`
  in `fastfields-lib`/`fastfields-cpu-lib`) and every other output-first raw
  binding (`sym_matvec`, `sym_solve`, `dt_euclidean`'s in-place write target,
  etc.) — it is consistent *within its own layer*.
- The three wrappers, by contrast, are free to allocate, so they present the
  pythonic, input-first `resample(inp, ...) -> out` signature every numpy/
  scipy user expects, and — this is the part that matters for `auto` — they
  agree with **each other**, which is the only agreement `fastfields.auto`
  actually depends on. `fastfields.auto` never touches `fastfields.dlpack`
  directly, so this divergence is invisible to it.

In short: the wrapper layer already achieved full unification (`inp` first, on
numpy, torch and cupy alike); the binding's `out`-first order is a distinct,
necessary consequence of not owning an allocator, not a contract gap.

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
- `sym_invert_` and `dt_euclidean_`/`dt_l1_` remain **out-of-place only** on
  torch, and for two different reasons — worth spelling out precisely, because
  they look alike but aren't:
  - `sym_invert` is genuinely **nonlinear** in the matrix (`d(inv(M))/dM`
    depends on `M`), so an in-place `sym_invert_` would need the pre-mutation
    matrix for backward. It isn't even differentiable today
    (`fastfields-torch/_sym.py` raises if `mat.requires_grad`) — the rule
    bites before the question of in-place even comes up.
  - `dt_euclidean`/`dt_l1` are **not differentiable at all**
    (`fastfields-torch/_dt.py` raises via `_reject_grad`), so there is no
    backward to protect. The wrapper always clones purely as a defensive
    default (never surprise-mutate a caller's tensor on a non-differentiable
    op), not because the additive-backward rule forces it. This is a
    deliberate, conservative choice, not the rule from the table above —
    see the fastfields#4 discussion for the reasoning.
- `spline_coeff_` and `sym_solve_` are two ops where the backward *as
  currently implemented* does not read the pre-mutation tensor either
  (`_SplineCoeff.backward` only needs the saved `spline`/`bound` scalars;
  `_Solve.backward` only needs the saved `mat`/`weight`, never `vec`) — so by
  the letter of the rule above they would qualify for an in-place form. They
  are kept out-of-place on torch anyway, as a deliberate, conservative
  choice: an in-place op still requires `ctx.mark_dirty()` machinery to
  protect against a *different* tensor in the graph aliasing the mutated
  buffer, and the memory saved is minor for the modest tensor sizes these ops
  are typically used on. Torch is not resource-constrained the way numpy/cupy
  buffers can be, so out-of-place-by-default was judged the safer stance.
  Flagged explicitly here (see fastfields#4) so a future contributor doesn't
  read the table above, "prove" these are safe, and add them without
  re-deriving this same tradeoff.

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

Because these live outside the canonical set, `fastfields.auto` never routes to a
`_`-suffixed op implicitly; they are backend-specific extensions.

Beyond the regulariser accumulate set, **numpy and cupy expose the identical
in-place surface for every non-differentiable-on-torch op**:
`dt_euclidean_`, `dt_l1_`, `spline_coeff_`, `sym_addmatvec_`, `sym_submatvec_`,
`sym_solve_`, `sym_invert_`. Neither backend has autograd, so nothing blocks
in-place there; earlier revisions had numpy spell some of these as an
`inplace=` keyword and were missing `sym_solve_`/`sym_invert_` outright —
that was an accidental gap (not a deliberate difference) and has been closed
(fastfields#4; see `fastfields-numpy` PR #24).

## Backend-specific extensions (allowed)

Beyond the canonical set, backends may add:

- **numpy** — `sym_matvec_backward`, `sym_channels_from_packed`,
  `dt_spline_{table,brent,gaussnewton}`, and the in-place set
  `dt_euclidean_`/`dt_l1_`/`spline_coeff_`/`sym_solve_`/`sym_invert_`
  (mirroring cupy's in-place surface — see "Availability" above).
- **torch** — the regulariser accumulate set
  (`{field,flow}_{add,sub}{matvec,diag}` and their `_` in-place forms),
  which are autograd-safe by the rule above. Nothing else beyond the
  canonical set.
- **cupy** — the regulariser accumulate set above, plus
  `dt_euclidean_`/`dt_l1_`/`spline_coeff_`/`sym_solve_`/`sym_invert_`
  (identical to numpy's in-place set), `current_stream_ptr`, and the
  `dt_spline_*` variants.

These are not part of the `auto` contract; code that relies on them targets a
specific backend on purpose.

## What the conformance test checks

`tests/test_conformance.py`:

1. **Surface** — every installed backend exposes each canonical name (and the
   `Spline`/`Bound` enums).
2. **In-place policy** — numpy exposes the full non-differentiable in-place
   set (`sym_addmatvec_`/`sym_submatvec_`/`sym_solve_`/`sym_invert_`/
   `dt_euclidean_`/`dt_l1_`/`spline_coeff_`); torch exposes none of those
   plus the regulariser accumulate set (the documented autograd exclusion).
3. **Equivalence** — for the shared ops, a numpy call and the equivalent torch
   call (dispatched through `fastfields.auto`) produce numerically identical
   results, so `auto` means the same thing on both. cupy is skipped where no GPU
   is available.
4. **`dt_mesh` positional compatibility** — `signed`/`naive`/`return_nearest`
   are not keyword-only on any backend, so a positional call cannot silently
   work on some backends and raise `TypeError` on others.
