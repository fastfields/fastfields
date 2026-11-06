# One distribution, five namespaces

`fastfields` used to be five separate PyPI distributions, one for each
subpackage of the `fastfields` namespace. It is now one distribution that ships
all five. The import surface did not move: the same `fastfields.<sub>` modules
exist with the same contents, and `fastfields.dlpack` still merges into the
same namespace from its own wheel.

| was | shipped | now |
|---|---|---|
| `fastfields-helpers` | `fastfields.helpers` | `fastfields` |
| `fastfields-numpy` | `fastfields.numpy` | `fastfields` |
| `fastfields-cupy` | `fastfields.cupy` | `fastfields` |
| `fastfields-torch` | `fastfields.torch` | `fastfields` |
| `fastfields` | `fastfields.auto` | `fastfields` |
| `fastfields-dlpack` | `fastfields.dlpack` | `fastfields-dlpack` (unchanged) |

## What stayed the same, on purpose

**The subpackages are still PEP 420 namespace subpackages.** There is no
`fastfields/__init__.py` anywhere, in this distribution or in
`fastfields-dlpack`, and that absence is the whole mechanism. `fastfields` is a
native namespace package, so two independently installed wheels can each
contribute a `fastfields/<sub>/` directory and both stay importable. Add a
top-level `__init__.py` and `fastfields` becomes an ordinary package owned by
whichever wheel shipped it, and `fastfields.dlpack`, the compiled half, stops
being importable. For this reason the merged `pyproject.toml` lists each
subpackage explicitly with `namespaces = true`, and
`tools/consolidate-python.sh` fails if an `__init__.py` ever appears there.

**`fastfields-dlpack` stays a separate distribution and repository.** It is the
only compiled one. It needs a per-platform wheel matrix, and
`libfastfields-cpu.so` alone is about 166 MB. Folding it in would mean that
every one-line Python fix rebuilds and re-uploads that whole matrix, which is
exactly why `fastfields-helpers` was split *out* of it in the first place.
Merging the five pure-Python distributions keeps that property: `fastfields` is
still one universal `py3-none-any` wheel with no compiled content, so a fix to
any of the five ships as an ordinary pure-Python release. There is no
circularity either, because `fastfields-dlpack` depends only on `packaging` and
never on anything here.

**The array libraries are still optional.** `numpy`, `torch` and `cupy` are
extras (`fastfields[numpy]`, `[torch]`, `[cupy]` and `[all]`), so
`pip install fastfields` works with whatever subset you have, and
`fastfields.auto` dispatches over exactly that subset. `fastfields-dlpack` is
the only hard dependency.

## The one wrinkle: an extra is not a lane

Read this before filing a bug.

`fastfields-dlpack` records the compute backend it was compiled for in the
local segment of its PEP 440 version, in the PyTorch style: `0.0.dev0+cpu`,
`+cu118`, `+cu126`, `+cu130`. At runtime that is
`fastfields.dlpack.backend`. **Which one you get is decided by the index folder
you install from**, not by the dependency graph:

```sh
pip install fastfields --extra-index-url https://fastfields.github.io/whl/cpu/
pip install fastfields --extra-index-url https://fastfields.github.io/whl/cu130/
```

No package in this distribution pins a lane, and none can. `pip` has no way to
express "any build with CUDA", and a pin would fight the `--extra-index-url`
you just chose. So the dependency on `fastfields-dlpack` is deliberately
unqualified.

The consequence is that **the `cupy` extra selects the CuPy client library, not
the CUDA lane**:

```sh
pip install "fastfields[cupy]"  # succeeds, on the CPU lane too
python -c "import fastfields.cupy"  # ImportError
```

That install is not wrong in any way `pip` can see, and it is not a resolution
failure. It is a mismatch that only the package itself can detect, so each
namespace enforces its own requirement at runtime:

| namespace | needs | what it does |
|---|---|---|
| `fastfields.numpy` | CPU | Nothing. The CPU path works on any build. |
| `fastfields.cupy` | CUDA | **Fails at import** if `fastfields.dlpack.backend` is not `cu*`. Every operation runs in CUDA device memory, so there is no partial mode worth offering. The message names the index URL and the `pip` command. |
| `fastfields.torch` | either | **Fails lazily.** The CPU path is the tested one and runs on any build, so the import always succeeds and the check happens only when a CUDA tensor arrives, in `stream_ptr`, which every operation already calls. |
| `fastfields.auto` | whatever is installed | Reports. `fastfields.auto.dlpack_backend()` returns the label, and dispatch is dynamic already. |

An unlabelled build (`backend is None`, which is what an unversioned source
checkout reports) is let through everywhere. It means *unknown*, not *CPU*, and
every wheel from the index carries a real label, so refusing it would block GPU
development from a source tree while catching no real mistake.

## How the merge was done

`tools/consolidate-python.sh` is the frozen record. It is the Python-side
sibling of `tools/consolidate.sh` in `fastfields-lib`. In outline there was one
`git filter-repo` path-move pass for each absorbed repository, followed by four
`--allow-unrelated-histories` merges into this repository's existing history.
This repository was not rewritten, so its `main` is still a true ancestor of
the result and every existing commit SHA and permalink against it still
resolves.

History is preserved across the old `fastfields_numpy/` to `fastfields/numpy/`
layout rename too, which the path rules fold onto one path so that
`git log --follow` crosses it. Issue and PR *numbers* do not transfer, because
GitHub does not redirect them across a repository merge. A reference to, say,
`fastfields-numpy#23` remains a reference to that archived repository.
