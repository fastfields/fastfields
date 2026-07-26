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

In-place (`_`-suffixed) variants mutate the caller's buffer and return it. Their
availability differs **intentionally**:

- **numpy / cupy** expose the in-place set (e.g. `sym_addmatvec_`,
  `sym_submatvec_`; cupy additionally `dt_euclidean_`, `sym_solve_`,
  `sym_invert_`, `spline_coeff_`, …). They mutate the caller's array in place.
- **torch omits every in-place op.** In-place mutation does not compose with
  autograd (it breaks the graph / gradient bookkeeping), so the torch backend is
  deliberately a smaller, functional-only surface. Callers that need in-place
  semantics should use numpy or cupy.

Because these live outside the canonical set, `fastfields.any` never routes to a
`_`-suffixed op implicitly; they are backend-specific extensions.

## Backend-specific extensions (allowed)

Beyond the canonical set, backends may add:

- **numpy** — `sym_matvec_backward`, `sym_channels_from_packed`,
  `dt_spline_{table,brent,gaussnewton}`.
- **torch** — nothing beyond the canonical set (autograd-focused, minimal).
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
