# fastfields  (imports as `fastfields.auto`)

The umbrella distribution. Provides **`fastfields.auto`**, a backend-dispatching
interface over the per-backend wrappers, and anchors the `fastfields` PEP 420
namespace the other distributions merge into. This is the top of the stack.

```
… ─ dlpack ─ {numpy,cupy,torch} ─ fastfields ← (you are here)
```

## Philosophy / role
- `fastfields.auto` exposes one unified function per operation and dispatches on
  the **type of the first array argument**:
  `numpy.ndarray -> fastfields.numpy`, `torch.Tensor -> fastfields.torch`,
  `cupy.ndarray -> fastfields.cupy`.
- **Lazy backend imports**: only the backend whose array type is passed is
  imported, so `fastfields.auto` works with whatever subset of numpy/torch/cupy
  is installed. Only `fastfields-dlpack` is a hard dependency; array backends
  come via extras (`fastfields[numpy|torch|cupy|all]`).
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
  **no `fastfields/__init__.py`** — the five installs
  (`dlpack`/`numpy`/`torch`/`cupy`/`auto`) merge into one importable `fastfields`
  namespace. Do not add a top-level `__init__.py`.
- Keep the unified→backend name mapping in sync when backend wrappers rename
  functions.
- Ruff: line-length 79, select B/E/F/I/W.

## Pointers
- Hierarchy: `/home/user/.github/profile/README.md`.
- Status: `/home/user/fastfields-lib/MIGRATION.md`.
