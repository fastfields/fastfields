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
| `dt_euclidean` | `(inp, voxel_spacing=1.0)` | Euclidean distance transform along the last axis. |
| `dt_l1` | `(inp, voxel_spacing=1.0)` | L1 distance transform. |
| `dt_mesh` | `(loc, vertices, faces, signed=True, naive=False, return_nearest=False)` | Point-to-triangular-mesh distance. `faces` is normalized to int64. `signed`/`naive`/`return_nearest` are positional-or-keyword on every backend (not keyword-only) — see fastfields#4. |
| `dt_spline_table` | `(loc, coeff, times, *, order=3, bound="dct2")` | Point-to-spline distance over a dictionary of candidate times. Returns **`(dist, time)`** — see below. |
| `dt_spline_brent` | `(loc, coeff, *, max_iter=64, tol=1e-6, step=0.1, order=3, bound="dct2")` | Same, by Brent's method. Returns **`(dist, time)`**. |
| `dt_spline_gaussnewton` | `(loc, coeff, *, max_iter=64, tol=1e-6, order=3, bound="dct2")` | Same, by Gauss-Newton. Returns **`(dist, time)`**. |
| `sym_matvec` | `(mat, vec)` | Compact-symmetric matrix-vector product. |
| `sym_solve` | `(mat, vec, weight=None)` | Solve `(mat + diag(weight)) x = vec`. |
| `sym_invert` | `(mat)` | Invert a compact-symmetric matrix. |
| `resample` | `(inp, factor=None, shape=None, *, order=2, bound="dct2", ndim=None, anchor="centers", shift=None, scale=None)` | Spline resample. `factor` **or** `shape` (mutually exclusive); a positional `2` is a **factor** on every backend. |
| `restriction` | same as `resample` | Adjoint of `resample` at reciprocal scale. |
| `spline_coeff` | `(inp, order=3, bound="dct2")` | Spline-coefficient prefilter along the last axis. |
| `Spline`, `Bound` | enums | Re-exported by every backend. |

Positional call `resample(inp, 2)` therefore means *factor 2* on numpy, torch and
cupy alike — the historical footgun (a bare `2` meaning an output shape on
torch/cupy) is gone (see fastfields-lib#17 C2).

Unless a parameter is explicitly marked keyword-only in the table above (the
`*` in `resample`'s signature), it is **positional-or-keyword on every
backend**. `dt_mesh`'s `signed`/`naive`/`return_nearest` used to be
keyword-only on numpy only, which meant a positional call worked on
torch/cupy and raised `TypeError` on numpy — an accidental divergence with no
technical justification, fixed in `fastfields-numpy` PR #24 (fastfields#4).

### `dt_spline_*` returns `(dist, time)`

The three point-to-spline ops are the one canonical operation that returns a
*tuple*, so the contract has to cover its order, and it is **`(dist, time)`**
in every namespace. `fastfields.cupy` returned `(time, dist)` until
fastfields#38.

That mattered more than an inconsistency: `fastfields.auto` forwards both
arguments and result unchanged, so it has nowhere to normalise a *result*
tuple, and the meaning of `fastfields.auto.dt_spline_table(...)`'s return
value depended on which array type the caller passed.

Two things worth being exact about:

- **The raw `fastfields.dlpack` binding takes `(time, dist, ...)`** as its two
  output buffers — the order the C++ hub declares, and not changing (same
  output-first reasoning as the next section). So in every wrapper the binding
  call and the `return` are deliberately in opposite orders, within adjacent
  lines of the same function. Aligning them would silently report distance as
  time, so `tests/test_dt_spline_return_order_cupy.py` pins both.
- **`tests/test_conformance.py` checks the names, not the order.** `import
  fastfields.cupy` refuses a CPU-only `fastfields-dlpack`, so the cupy
  parametrisations of that suite skip wherever CI runs; the order is pinned
  statically instead. The names are nonetheless canonical now — their absence
  from `CANONICAL` is how the divergence survived to need an issue.

**Signatures now agree too.** They did not when fastfields#38 landed:
`fastfields.cupy` spelled these `(loc, coeff, times, spline=3, bound=3)`,
with `max_iter`/`tol`/`step` *required and positional*. cupy has moved to the
numpy/torch signature in the table — `order` keyword-only with a default,
`bound` taking the friendly `"dct2"` spelling, and the three refiner knobs
keyword-only with the same defaults the other two backends use.

`order` rather than `spline` deserves a word, because the C++ layer says
`spline`: every `ff::` hub entry point spells the spline order `spline`
(`int8_t spline = spline_t::Cubic`, 15 declarations, no `order` anywhere),
and the raw `fastfields.dlpack` binding keeps that name. But the **wrapper**
layer settled on `order` long ago and settled on it everywhere —
`resample`, `restriction`, `spline_coeff`, `pull`, `push`, `count`, `grad`,
on all three backends, cupy included. cupy's `dt_spline_*` was the single
place in the Python surface that still said `spline`, so aligning it to
`spline` would have meant renaming `order` in nine other operations across
three backends, against this table, to chase a name the binding layer owns
on purpose (see "Why the raw `fastfields.dlpack` binding signature differs"
below). Two of three backends, this table's canonical row, and the wrapper
layer's own vocabulary all say `order`.

## Buffer-argument naming

`fastfields.auto` forwards `**kwargs` unchanged, so the *name* of an
operation's first argument is part of the contract, not an internal detail:
if one backend calls it `inp` and another `x`, then
`fastfields.auto.dt_euclidean(inp=a)` works for numpy arrays and raises
`TypeError` for torch tensors, or the reverse. Three conventions, applied
across all three backends:

| First argument | Spelled | Examples |
| --- | --- | --- |
| the input array, not modified | `inp` | `dt_euclidean`, `dt_l1`, `resample`, `restriction`, `spline_coeff`, `pull`, `grad`, the `{field,flow}_*` accumulate ops |
| a buffer read **and** written in place | `inp_out` | `dt_euclidean_`, `dt_l1_`, `spline_coeff_`, `sym_solve_` |
| an accumulator written through, whose prior contents are an input | `out0` | `sym_addmatvec_`, `sym_submatvec_` |

`inp_out` is the hub's own name for a read-and-written tensor
(`void dt_euclidean(DLTensor & inp_out, ...)`, likewise `spline_coeff`), and
`sym_solve_` already used it on all three backends; `dt_euclidean_`/`dt_l1_`/
`spline_coeff_` were the stragglers (`x` on numpy, `inp` on torch). The
compact-symmetric operands are `mat`/`vec`/`grad` in the wrapper layer, where
the hub says `hessian`/`inp`/`grd` — cupy had been following the hub and now
follows the wrappers, for the same reason `order` won above.

## Where a backend's signature may legitimately differ

Two exemptions, and only two. Both are enforced by name in
`tests/test_signature_conformance.py` rather than inferred, so widening them
is a reviewable edit.

- **`dtype` defaults.** `{field,flow}_{diag,kernel}` and
  `{field,flow}_diag_rls` allocate their own output and so take a `dtype`.
  Its default cannot be spelled portably: `np.float64` on numpy,
  `torch.float64` on torch, and `None` on cupy — because
  `fastfields.cupy` imports cupy lazily and cannot name `cupy.float64` at
  module scope, so the body resolves `dtype or cp.float64`. All three mean
  float64. The parameter's name, kind and position are still identical on
  every backend; only the default *expression* differs.
- **`device` on torch.** A trailing keyword-only parameter with a default, on
  the same six builders. numpy and cupy have no device to choose. This is the
  golden rule working as intended: a call written against the shared
  signature keeps working on torch.

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
- `spline_coeff_` and `sym_solve_` **are** exposed on torch, and are fully
  differentiable in place: their backward — identical to the out-of-place
  form's — never reads the pre-mutation tensor (`_SplineCoeff.backward` only
  needs the saved `spline`/`bound` scalars; `_Solve.backward` only needs the
  saved `mat`/`weight`, never `vec`), so mutating in place destroys no
  information backward needs. This satisfies the per-op rule above cleanly;
  the earlier revision that kept them out-of-place-only "for autograd
  reasons" had no such reason once you check the actual backward — that was
  the mislabeled half of the fastfields#4 concern, now fixed and gradcheck-
  verified (see `fastfields-torch` tests).

### Non-differentiable ops: exposed everywhere, backward raises (not omitted)

`dt_euclidean`, `dt_l1`, `dt_mesh` and `sym_invert` have **no supported
gradient** — for `dt_euclidean`/`dt_l1`/`dt_mesh` because the underlying
op has no meaningful gradient at all; for `sym_invert` because the inverse
is nonlinear in the matrix (`d(inv(M))/dM` depends on `M`) and no backward
is implemented for it on any backend. This is a **different** situation
from the additive-vs-nonlinear in-place question above — it's about whether
the *op itself* has a gradient, not about whether mutating in place is safe.

**Earlier revisions handled this two different, both-wrong ways**: the
in-place forms (`dt_euclidean_`, `dt_l1_`, `sym_invert_`) were omitted from
torch entirely — described as excluded "for autograd reasons", as if a
gradient existed that in-place mutation would corrupt, when in fact there
was no gradient to protect in the first place. The out-of-place forms were
kept, but guarded by rejecting any grad-requiring input *at call time*
(`ValueError` from `forward`, before a graph even existed) — which meant a
call could fail even when the caller never intended to backprop through it,
and the failure mode differed from every other non-differentiable case in
the ecosystem (e.g. `torch.argmax`, which lets the graph form and simply
carries no gradient).

**The current policy, on every backend**:

- All four ops — and their in-place forms where one exists
  (`dt_euclidean_`, `dt_l1_`, `sym_invert_`; `dt_mesh` has no in-place form
  on any backend, since its output shape/target differs from every input) —
  are exposed, for full parity with numpy/cupy. No op is omitted because it
  lacks a gradient.
- On torch specifically, forward always runs normally, including when an
  input requires grad, so the output can sit inside a larger autograd graph
  (e.g. as an intentional stop-gradient boundary). Only calling
  `.backward()` through that output raises — a `torch.autograd.Function`
  whose `backward()` raises a clear `RuntimeError` naming the op and stating
  it has no gradient (see `fastfields.torch._util.raise_not_differentiable`
  and each function's docstring), never a generic/cryptic autograd internal
  error and never a silent wrong answer.
- The in-place forms additionally call `ctx.mark_dirty()` like every other
  in-place `torch.autograd.Function` here, and are subject to the same
  leaf rule as any in-place op.
- numpy/cupy have no autograd, so their `_`-suffixed forms
  (`dt_euclidean_`, `dt_l1_`, `sym_invert_`) simply mutate and return —
  there is nothing to guard on those backends.

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

Beyond the regulariser accumulate set, **numpy, torch and cupy now all expose
the same in-place surface**: `dt_euclidean_`, `dt_l1_`, `spline_coeff_`,
`sym_solve_`, `sym_invert_`, `sym_addmatvec_`, `sym_submatvec_`. Neither numpy
nor cupy has autograd, so nothing blocks in-place there; earlier revisions had numpy
spell some of these as an `inplace=` keyword and were missing
`sym_solve_`/`sym_invert_` outright — that was an accidental gap (not a
deliberate difference) and has been closed (fastfields#4; see
`fastfields-numpy` PR #24). Torch was missing all five outright — also
closed (fastfields#4; see `fastfields-torch`'s "Non-differentiable ops"
section above and its "in-place autograd-safe" fixes for `sym_solve_`/
`spline_coeff_`).

`sym_addmatvec_`/`sym_submatvec_` (the posdef accumulate ops, distinct from
the regulariser `{field,flow}_{add,sub}matvec_` family above) **are on torch
too**, as of fastfields-torch#35; an earlier revision of this section
described them as numpy/cupy-only with no torch wrapper at all, which stopped
being true when that landed. There is still no out-of-place
`sym_addmatvec`/`sym_submatvec` on any backend — the C primitive is in-place
only, and unlike the regulariser accumulate pair these were never given the
clone-then-accumulate spelling (see "One kernel, two spellings"). That is the
remaining asymmetry, and it is the same on all three backends rather than a
cross-backend divergence.

## Backend-specific extensions (allowed)

Beyond the canonical set, backends may add:

- **numpy** — `field_cg` (+ the `CGInfo` record it returns; the CG driver is
  CPU-only in `fastfields-lib`, see fastfields-lib#34),
  `sym_channels_from_packed`, `sym_matvec_backward`,
  `field_precond_`/`flow_precond_`, and the in-place set
  `dt_euclidean_`/`dt_l1_`/`spline_coeff_`/`sym_solve_`/`sym_invert_`/
  `sym_addmatvec_`/`sym_submatvec_`.
- **torch** — the regulariser accumulate set
  (`{field,flow}_{add,sub}{matvec,diag,kernel}` and their `_` in-place forms),
  which are autograd-safe by the rule above; and the in-place set
  `dt_euclidean_`/`dt_l1_`/`spline_coeff_`/`sym_solve_`/`sym_invert_`/
  `sym_addmatvec_`/`sym_submatvec_` (mirroring numpy/cupy — see
  "Availability" above). `spline_coeff_`/`sym_solve_` are differentiable;
  `dt_euclidean_`/`dt_l1_`/`sym_invert_` raise a clear `RuntimeError` from
  `.backward()` (see "Non-differentiable ops" above). Nothing else beyond the
  canonical set — in particular, no `field_cg` and no
  `field_precond_`/`flow_precond_`.
- **cupy** — the regulariser accumulate set above, plus `sym_matvec_backward`,
  `field_precond_`/`flow_precond_`,
  `dt_euclidean_`/`dt_l1_`/`spline_coeff_`/`sym_solve_`/`sym_invert_`/
  `sym_addmatvec_`/`sym_submatvec_` (identical to numpy's in-place set) and
  `current_stream_ptr`.

That list is the presence matrix, and
`tests/test_signature_conformance.py::test_partial_op_is_on_exactly_the_documented_backends`
asserts it **exactly**, in both directions: a backend that loses one of these
fails, and so does a backend that quietly gains one — which is how the
"`sym_addmatvec_` has no torch wrapper at all" claim above managed to go stale
for a release cycle.

These are not part of the `auto` contract; code that relies on them targets a
specific backend on purpose. In particular, `fastfields.auto` never routes to
any `_`-suffixed op implicitly, torch's non-differentiable ops included —
calling one through `auto` still means calling it directly on the resolved
backend and taking on that backend's differentiability behavior yourself.

## What the conformance test checks

`tests/test_conformance.py`:

1. **Surface** — every installed backend exposes each canonical name (and the
   `Spline`/`Bound` enums).
2. **In-place / non-differentiable-op parity** — numpy, torch and cupy all
   expose the same `dt_euclidean_`/`dt_l1_`/`spline_coeff_`/`sym_solve_`/
   `sym_invert_` in-place surface (no backend silently omits one of these);
   `sym_addmatvec_`/`sym_submatvec_` remain numpy/cupy-only (torch has no
   `sym_addmatvec`/`sym_submatvec` at all, see "Availability" above); torch
   additionally exposes the regulariser accumulate set. On torch, calling
   `.backward()` through `dt_euclidean`/`dt_euclidean_`/`dt_l1`/`dt_l1_`/
   `sym_invert`/`sym_invert_` raises a `RuntimeError` naming the op.
3. **Equivalence** — for the shared ops, a numpy call and the equivalent torch
   call (dispatched through `fastfields.auto`) produce numerically identical
   results, so `auto` means the same thing on both. cupy is skipped where no GPU
   is available.
4. **`dt_mesh` positional compatibility** — `signed`/`naive`/`return_nearest`
   are not keyword-only on any backend, so a positional call cannot silently
   work on some backends and raise `TypeError` on others.

That suite checks **names**, and it checks them by importing. Both limits
matter, and `tests/test_signature_conformance.py` exists because of them:

- A name check cannot see a divergence in *parameters*, *defaults* or
  *return order* — which is the whole of what cupy's `dt_spline_*` got wrong
  while every name assertion passed.
- `import fastfields.cupy` deliberately refuses a CPU-only
  `fastfields-dlpack`, which is the build CI has, so every cupy
  parametrisation of an importing suite **skips** — in exactly the
  environment that has to catch a cupy divergence. A test that skips in CI
  catches nothing.

So `tests/test_signature_conformance.py` imports none of numpy, torch, cupy or
`fastfields.dlpack`. It finds each `fastfields/<backend>/` through the
namespace package's search locations and parses the sources with `ast` — the
pattern `tests/test_dt_spline_return_order_cupy.py` and
`tests/test_reg_family_routing_cupy.py` already use. It cannot skip, and it
reports all three backends on a CPU runner with no GPU and no torch
installed. It checks:

1. **Signature agreement** — for every operation all three backends export,
   the parameter names, kinds (positional-or-keyword vs keyword-only), order
   and defaults are identical, modulo the two exemptions in "Where a
   backend's signature may legitimately differ". A parameter only one backend
   has must be declared in `BACKEND_ONLY_PARAMS` there, and must be
   keyword-only with a default.
2. **The `dt_spline_*` family by value** — the three signatures are pinned
   literally, so "align all three onto the wrong spelling" fails as loudly as
   leaving one behind.
3. **Buffer-argument naming** — the `inp` / `inp_out` / `out0` conventions
   above, checked on the first parameter of each op that has one. `auto`
   forwards `**kwargs`, so these names are contract.
4. **The presence matrix** — `PARTIAL_OPS` lists every operation that is
   deliberately on a subset of backends, and is asserted exactly; everything
   else must be on all three.
5. **`fastfields.auto`'s routing tables** — read out of the `_ALIASES` /
   `_DISPATCHED` / `_PARTIAL` literals and cross-checked against what the
   backends actually define, both ways. `make_dispatcher` resolves
   `getattr(backend, attr)` at *call* time, so a stale table entry would
   otherwise surface as a `NotImplementedError` inside a user's call rather
   than as a red CI job. Every operation all three backends expose must have
   a dispatcher, or be listed in `AUTO_EXEMPT` with the reason it has no
   array argument to dispatch on.
6. **Return tuples** — numpy and cupy return plain local names, so their
   return *order* is readable from the source and is required to match.
   (torch returns `_SomeFunction.apply(...)`, whose order lives in that
   `Function`'s `forward`; that stays covered at runtime by
   `tests/test_new_surface_torch.py`, which can import torch in CI.)
