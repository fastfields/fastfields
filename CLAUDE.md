# fastfields  (imports as `fastfields.auto`)

The umbrella distribution. Provides **`fastfields.auto`**, a backend-dispatching
interface over the per-backend wrappers, and anchors the `fastfields` PEP 420
namespace the other distributions merge into. This is the top of the stack.

```
helpers ─┐
dlpack ─ {numpy,cupy,torch} ─ fastfields ← (you are here)
```

## Philosophy / role
- `fastfields.auto` exposes one unified function per operation and dispatches on
  the **type of the first array argument**:
  `numpy.ndarray -> fastfields.numpy`, `torch.Tensor -> fastfields.torch`,
  `cupy.ndarray -> fastfields.cupy`.
- **Lazy backend imports**: only the backend whose array type is passed is
  imported, so `fastfields.auto` works with whatever subset of numpy/torch/cupy
  is installed. Only `fastfields-helpers` -- the pure-Python enums/normalisers,
  no compiled extension -- is a hard dependency, so `pip install fastfields`
  alone pulls no compiled wheel; array backends (which carry their own
  `fastfields-dlpack` dependency) come via extras
  (`fastfields[numpy|torch|cupy|all]`). `fastfields.auto` itself never imports
  `fastfields.dlpack` -- verified by grep across this repo, only
  `fastfields/auto/__init__.py` ever touched it, and that import is now gone.
- Every backend exposes the same canonical `dt_*` / `sym_*` names, and each
  unified name (e.g. `dt_euclidean`, `dt_mesh`) maps straight onto that function
  in the selected backend. Args are forwarded unchanged, so a dispatched call
  shares the selected backend's signature.
- Raises a clear `TypeError` for an unknown array type and a clear `ImportError`
  when the needed backend isn't installed.

## Layout
`fastfields/auto/`: `__init__.py`, `_dt.py`, `_sym.py`, `_resample.py`,
`_util.py` (the dispatch machinery). `tests/test_auto.py`.

## Build & test
```
pip install .                    # or pip install "fastfields[all]"
python -m pytest tests/ -q       # import from a neutral cwd
```
Prefer a regular install over editable (native-namespace merge).

## Conventions & caveats
- **PEP 420 namespace**: this distribution ships only `fastfields/auto/` and
  **no `fastfields/__init__.py`** — the six installs
  (`helpers`/`dlpack`/`numpy`/`torch`/`cupy`/`auto`) merge into one importable
  `fastfields` namespace. Do not add a top-level `__init__.py`.
- Keep the unified→backend name mapping in sync when backend wrappers rename
  functions. `tests/test_auto_surface.py` is the forcing function: it
  parametrises over every `_ALIASES` entry in every category module and
  asserts the named attribute exists on that backend, so a table that drifts
  from the wrappers fails rather than waiting for a call.
- **Dispatch failures must be intelligible.** `_util.make_dispatcher` used to
  end in a bare `getattr(backend, attr)(*args, **kwargs)`, so a table that
  over-promised produced an `AttributeError` naming a module the caller never
  mentioned. A mixed install is ordinary here -- the five distributions are
  versioned and released independently -- so a missing attribute now raises
  `NotImplementedError` saying which package to upgrade, and an op a backend
  genuinely lacks raises one naming the backends that do have it
  (`_reg._PARTIAL`).
- **Two kinds of op are deliberately not dispatchable**, and both are spelled
  out in the package docstring (and checked by `test_auto_surface.py`):
  shape-taking builders (`{field,flow}_{diag,kernel}` -- no array to dispatch
  on) and the pushpull adjoints (`*_backward` -- autograd-only, wired into
  torch's `autograd.Function`).
- **Known cross-backend inconsistency**: the three `dt_spline_*` ops return
  `(dist, time)` on numpy/torch and `(time, dist)` on cupy. A dispatcher
  cannot normalise a *result*, so this is documented in `_dt.py` and
  `docs/index.md` and needs a decision upstream.
- Ruff: line-length 79, select B/E/F/I/W.

## Pointers
- Hierarchy: `/home/user/.github/profile/README.md`.
- Status: `/home/user/fastfields-lib/MIGRATION.md`.
