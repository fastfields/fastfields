# fastfields — the Python side

One distribution, `fastfields`, shipping five PEP 420 subpackages of the
`fastfields` namespace: **`fastfields.{helpers,numpy,cupy,torch,auto}`**.
Pure Python, one universal `py3-none-any` wheel, no compiled content.

```
fastfields-lib (C++/CUDA) ─ fastfields-dlpack ─┐
                                               ├─ fastfields.{numpy,cupy,torch} ─ fastfields.auto
                             fastfields.helpers ┘            (all in THIS repo)
```

On **2026-11-04** five repositories — `fastfields` (this one),
`fastfields-helpers`, `fastfields-numpy`, `fastfields-cupy`,
`fastfields-torch` — were merged here, and the other four are to be archived.
This repo's own history was **not** rewritten: its `main` is still a true
ancestor of the merged history, so every pre-merge commit SHA and permalink
against it still resolves. The four absorbed histories were moved into place
with `git filter-repo` and merged in with `--allow-unrelated-histories`; see
[`tools/consolidate-python.sh`](tools/consolidate-python.sh), the frozen
record, and [`docs/monorepo.md`](docs/monorepo.md).

> Issue and PR **numbers do not transfer across a repository merge** — GitHub
> does not redirect them. A reference like `fastfields-numpy#23` still points
> at that archived repo, and a bare `#23` in a commit message predating the
> merge means whatever `#23` meant in *its original* repo, not this one.

## `fastfields-dlpack` stays separate — do not fold it in

It is the only compiled distribution: a per-platform wheel matrix, and
`libfastfields-cpu.so` alone is ~166 MB. Folding it in would make every
one-line Python fix rebuild and re-upload that whole matrix — which is exactly
why `fastfields-helpers` was split *out* of it. Merging only the five
pure-Python distributions preserves that: a fix to any of them still ships as
an ordinary universal wheel. There is no circularity risk either —
`fastfields-dlpack` depends only on `packaging`.

## The PEP 420 rule — read before touching packaging

**There is no `fastfields/__init__.py`, anywhere, and creating one breaks
everything.** `fastfields` is a *native namespace package*: this
distribution's wheel and `fastfields-dlpack`'s wheel each contribute
`fastfields/<sub>/` directories, and only the absence of a top-level
`__init__.py` lets both be importable. Add one and `fastfields` becomes an
ordinary package owned by whichever wheel shipped it, and `fastfields.dlpack`
stops importing.

`pyproject.toml` lists the five subpackages explicitly under
`[tool.setuptools.packages.find]` with `namespaces = true` — without that flag
the find directive returns *nothing*, because none of the five has a parent
`__init__.py`. `tools/consolidate-python.sh` hard-fails if an `__init__.py`
appears at `fastfields/`.

This breaks at **import** time, not at build time, which is why the suite
installs the wheel and imports all six subpackages rather than trusting the
build.

## One version, one `_version.py`

versioningit writes exactly one file, so the merged distribution has one:
`fastfields/helpers/_version.py` (gitignored, written at build time).
`fastfields.{helpers,numpy,cupy,torch,auto}.__version__` all read it.

It is in `helpers` because that is the bottom of this distribution's internal
stack — the other four import it, it imports none of them — so reading the
version can never pull in an array library, and `auto` could not have held it
without a cycle. `fastfields.dlpack.__version__` belongs to a different
distribution and may legitimately differ.

## Compute backend: extras are not lanes

`fastfields-dlpack` records the backend it was compiled for in its version's
PEP 440 **local** segment (`0.0.dev0+cpu`, `+cu118`, `+cu126`, `+cu130`) and
exposes it as `fastfields.dlpack.backend`. **The lane is chosen by the index
folder the user installs from, not by the dependency graph** — `pip` cannot
express "any build with CUDA" — so the dependency here is deliberately
unqualified (`fastfields-dlpack==0.*`, a version bound that says nothing about
the local segment). **Never pin a variant.**

The cost is that `fastfields[cupy]` installs the *CuPy client*, not the CUDA
lane, and resolves happily on `cpu`. Each namespace therefore enforces its own
requirement at runtime:

| namespace | needs | enforcement |
|---|---|---|
| `fastfields.numpy` | CPU | none — the CPU path works on any build |
| `fastfields.cupy` | CUDA | **eager**: `fastfields/cupy/_backend.py`, the first local import in its `__init__`, raises `ImportError` on a non-CUDA label. Everything here runs in device memory, so there is no partial mode worth offering |
| `fastfields.torch` | either | **lazy**: `_util._require_cuda_backend`, called from `stream_ptr` — the one helper every op already calls with its primary tensor, and only on the CUDA branch. The import must keep working everywhere, because the CPU path is the tested one |
| `fastfields.auto` | whatever is installed | reports only: `fastfields.auto.dlpack_backend()` |

`backend is None` is **allowed** everywhere: it means *unknown* (an
unversioned source checkout), not *CPU*. Every wheel from the index carries a
real label, so refusing `None` would block GPU work from a source tree and
catch no real mistake. The message both checks raise is built by
`fastfields.helpers.cuda_backend_message`, which is a pure function of its
arguments — `fastfields.helpers` must never import `fastfields.dlpack`, so
each caller reads `backend` itself and passes it in.

## Per-subpackage notes

- **`fastfields.helpers`** — the shared layer; **stdlib only**, no numpy/
  torch/cupy and no `fastfields.dlpack`, ever. Enforced by
  `tests/test_helpers.py::test_import_is_stdlib_only`, which runs in a
  subprocess: in-process it would answer on test *ordering*, since the sibling
  suites in the same directory import numpy and torch at collection.
  Public surface: `Spline`, `Bound`, `as_spline`, `as_bound`,
  `normalize_shape`, `infer_ndim`, `check_ndim`, `resolve_out_spatial`,
  `anchor_scale_shift`, plus `is_cuda_backend` / `cuda_backend_message`.
  `normalize_shape` has no external caller but is deliberately public.
- **`fastfields.numpy`** — the reference wrapper. Functions return a fresh
  array (inputs untouched unless `inplace=True`); adds validation, zero-copy
  batch broadcasting and output allocation. Known caveat: the
  point-to-spline / point-to-mesh distance ops have inconsistent shape
  contracts between the C++ validation checks and kernels and are **not**
  covered by tests — see `fastfields-lib`'s `MIGRATION.md`.
- **`fastfields.cupy`** — the same surface on CuPy arrays in device memory,
  zero-copy via `__dlpack__`. Trailing-underscore wrappers write in place and
  never silently copy. Every wrapper forwards
  `cupy.cuda.get_current_stream().ptr` as the binding's `stream`, so kernels
  are ordered against surrounding CuPy work. **CuPy itself is imported
  lazily** (the right wheel depends on the host CUDA toolkit) — that is a
  separate axis from the CUDA-build check above, and the two are deliberately
  treated differently: a missing CuPy does not block the import, a CPU-only
  build does.
- **`fastfields.torch`** — autograd-enabled, structure mirroring `jitfields`.
  Differentiable: `sym_matvec`, `sym_solve`, `sym_solve_`, `resample`,
  `restriction`, `spline_coeff`, `spline_coeff_`. Exposed but not
  differentiable (parity with numpy/cupy): `dt_euclidean{,_}`, `dt_l1{,_}`,
  `dt_mesh`, `sym_invert{,_}` — forward always runs, and only `.backward()`
  through the output raises, via `_util.raise_not_differentiable`. See
  `API_CONTRACT.md` ("In-place policy").
- **`fastfields.auto`** — dispatches on the **type of the first array
  argument**, importing only the backend that type needs. Keep the
  unified→backend name tables in `_dt`/`_sym`/`_resample`/`_pushpull`/`_reg`
  in sync when a wrapper renames a function; a name missing from the selected
  backend now raises a message naming the op, the backend and the likely
  cause, instead of a bare `AttributeError`.

## Op naming

Renamed from the C++ impl layer and fixed across all five: `resize ->
resample`, `restrict -> restriction`, `splinc -> spline_coeff`.
`restriction` **accumulates into `out`**, so callers pre-zero it.

## Build & test

```bash
pip install fastfields-dlpack \
    --extra-index-url https://fastfields.github.io/whl/cpu/
pip install -e ".[test]"
cd /tmp && python -m pytest <repo>/tests -q    # matches the CI invocation
ruff check . && ruff format --check .          # line-length 79, B/E/F/I/W
codespell
```

Tests run **out of tree** deliberately, so the suite exercises the *installed*
package rather than the adjacent source directory — with a PEP 420 namespace
those are not the same thing. Python >= 3.9, so
`from __future__ import annotations` is what makes the `int | str`
annotations work.

`tests/` is one flat suite with unique module names — pytest's default import
mode keys modules on basename, so two `test_reg_family_routing.py` files
cannot coexist; the three that existed are suffixed per backend.

`tests/test_reg_family_routing_cupy.py`'s static guard **parses**
`fastfields/cupy/{__init__,_reg}.py` instead of importing them, on purpose:
`import fastfields.cupy` fails on the CPU-only build CI has, so importing
would silently skip the guard in the only environment where it is the only
guard there is.

## CI

Four workflows, all thin: `test.yaml`, `lint.yaml` and `release.yaml` call the
org-level reusable workflows in [`fastfields/.github`](https://github.com/fastfields/.github),
and `docs.yaml` builds the zensical site to GitHub Pages. `test.yaml` used to
be a ~60-line bespoke copy that installed the sibling wrapper repos over
`git+https`; all five wrappers are here now, so the suites test each other's
*source* and the copy is gone.

Releases are release-driven: publish a GitHub Release (tag e.g. `0.0.dev0`)
and `release.yaml` builds the wheel + sdist at that tag, uploads them, and
pings [`fastfields/whl`](https://github.com/fastfields/whl) to regenerate the
wheel index. `fetch-depth: 0` is **required** in every checkout —
versioningit derives the version from `git describe`, and a shallow checkout
hides the tags and silently yields `0+unknown`.

## Pointers

- [`docs/monorepo.md`](docs/monorepo.md) — the merge, what it preserved, and
  the extras-vs-lanes wrinkle in full.
- [`tools/consolidate-python.sh`](tools/consolidate-python.sh) — the frozen
  path-rewrite rules, with a warning about re-running them.
- [`API_CONTRACT.md`](API_CONTRACT.md) — the cross-backend surface contract
  (in-place policy, differentiability).
- Wheel index registration: `fastfields/whl`'s `sources.toml`.
- C++ status matrix and the full bug list:
  `/home/user/fastfields-lib/MIGRATION.md`.
- Hierarchy: `/home/user/.github/profile/README.md`.
