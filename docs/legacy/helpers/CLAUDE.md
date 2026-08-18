# fastfields-helpers

The **shared pure-Python layer**. Ships `fastfields.helpers`: the `Spline` /
`Bound` enums and the argument-normalisation helpers that every Python-facing
fastfields package uses. **Zero dependencies** — stdlib only, no compiled
component.

```
kernels ─ … ─ lib ─ bind-py ─ fastfields-dlpack ┐
                                                 ├─ {numpy,cupy,torch} ─ fastfields
fastfields-helpers  ← (you are here) ───────────┘
```

It sits *beside* the compiled stack, not on top of it. `fastfields-helpers` and
`fastfields-dlpack` are independent siblings: **neither depends on the other**,
and nothing here may ever import `fastfields.dlpack`.

## Philosophy / role
- One home for the backend-agnostic argument handling, so `fastfields.numpy`,
  `fastfields.torch`, `fastfields.cupy` and `fastfields.auto` share one
  implementation instead of each carrying a copy.
- **The dependency-free property is the whole point.** These helpers used to
  live in `fastfields-dlpack/fastfields/dlpack/_helpers.py`, alongside the
  compiled nanobind extension, which meant a one-line fix to a string alias or
  a rounding rule could only reach users through a full compiled-wheel release
  cycle built for every platform. Split out, they ship as a single universal
  wheel. Adding *any* runtime dependency here gives that back — don't.
- Correspondingly: no numpy/torch/cupy, no `fastfields.dlpack`, no compiled
  anything. `tests/test_helpers.py::test_import_is_stdlib_only` guards this.

## Layout
- `fastfields/helpers/__init__.py` — the entire public surface, one module.
  (`_version.py` sits beside it, written by versioningit at build time and
  gitignored.)
- `tests/test_helpers.py` — the only suite; ported from `fastfields-dlpack`'s
  `tests/test_bindings.py::test_shared_helpers` at the split.
- `docs/`, `zensical.toml`, `.github/workflows/{test,lint,release,docs}.yaml`.

## Public API (9 names)
- `Spline` — IntEnum, interpolation orders 0–7.
- `Bound` — IntEnum, 8 boundary conditions.
- `as_spline` / `as_bound` — `int` | enum | friendly string → `int`, via
  `_SPLINE_ALIASES` / `_BOUND_ALIASES`.
- `normalize_shape` — scalar-or-sequence → `list[int]` of length `ndim`.
- `infer_ndim` — infer spatial ndim from `ndim`/`factor`/`shape`.
- `check_ndim` — range validation.
- `resolve_out_spatial` — `factor`/`shape` → output spatial shape.
- `anchor_scale_shift` — torch-interpol anchor → (per-dim scale, scalar shift),
  via `_ANCHORS` / `_ANCHOR_SHIFT`.

`normalize_shape` currently has no external caller (only `resolve_out_spatial`
uses it) but is deliberately part of the documented public surface.

## Build & test
```
pip install ".[test]"
cd /tmp && python -m pytest <repo>/tests -q     # matches the CI invocation
ruff check . && ruff format --check .
```
Tests run out-of-tree deliberately, mirroring the reusable workflow, so the
suite exercises the *installed* package rather than the source directory it
happens to sit next to.

## Conventions & caveats
- **Python >= 3.9**; `from __future__ import annotations` is what lets the
  `int | str` annotations work there.
- `fastfields` is a **PEP 420 implicit namespace package** — there is
  deliberately **no** `fastfields/__init__.py`, which is what lets this package
  merge at import time with `fastfields.dlpack` / `.numpy` / `.torch` /
  `.cupy` / `.auto`. Creating one would break every sibling.
- Versioning is `versioningit`, the plain variant the pure-Python siblings use.
  Do **not** copy `fastfields-dlpack`'s `tool2version` / `_versioning.py`
  hooks: those stamp a backend label (`+cpu`, `+cu128`) into compiled wheels,
  and this package ships one unlabelled universal wheel.
- CI calls the org-level reusable workflows. `test.yaml` passes
  **`needs-dlpack: false`** (added in fastfields/.github#10) so the shared
  workflow skips its fastfields-dlpack source install and the clang toolchain
  steps — otherwise every runner would build the whole C++ chain to test
  stdlib-only code.
- Releases are release-driven: publish a GitHub Release (tag e.g.
  `0.0.0.dev1`) and `release.yaml` builds the wheel + sdist at that tag,
  uploads them, and pings `fastfields/whl`.

## Pointers
- Wheel index registration: `fastfields/whl`'s `sources.toml`.
- Hierarchy: `/home/user/.github/profile/README.md`.
