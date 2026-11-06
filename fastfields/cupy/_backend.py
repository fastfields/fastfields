"""Import-time guard: ``fastfields.cupy`` requires a CUDA build.

Importing this module raises `ImportError` when the installed
``fastfields-dlpack`` was not built for a CUDA backend. It is the first thing
[`fastfields.cupy`][fastfields.cupy]'s ``__init__`` imports, so the failure
happens at ``import fastfields.cupy`` rather than deep inside the first wrapper
call.

!!! note "Why this check exists at all"
    ``fastfields-dlpack`` ships the compiled kernels, and which compute backend
    it was built for is recorded in its version's PEP 440 *local* segment,
    PyTorch-style (``0.0.dev0+cpu``, ``+cu118``, ``+cu126``, ``+cu130``) and
    exposed as ``fastfields.dlpack.backend``. The lane is selected by the
    wheel-index folder the user installs from -- ``pip`` cannot express "any
    build with CUDA", so no package here pins one, and a pin would in any case
    fight the user's ``--extra-index-url`` choice.

    The consequence is that nothing in the dependency graph stops this
    combination: ``pip install fastfields[cupy]`` resolves perfectly happily
    against a ``+cpu`` ``fastfields-dlpack``, because the extra selects the
    *cupy client library*, not the wheel lane. Every operation here hands cupy
    device pointers to kernels that, in a CPU-only build, do not exist. So this
    is the one namespace of the five that must refuse to load.

!!! note "Why it is eager here and lazy in ``fastfields.torch``"
    Everything in ``fastfields.cupy`` operates on CUDA device memory -- there
    is no CPU path to fall back to, so there is nothing a CPU-only build could
    usefully do and no reason to defer the error. ``fastfields.torch`` is the
    opposite: its CPU path is the tested one and works on any build, so it must
    import cleanly and complain only when a CUDA tensor actually meets a
    CPU-only build (see ``fastfields/torch/_util.py``).

!!! note "Why an unlabelled build is allowed through"
    ``backend is None`` means *unknown*, not *CPU*: it is what an unversioned
    source checkout reports, since the label comes from the version string. Any
    wheel from the index carries a label, so refusing ``None`` would block GPU
    development from a source build while catching no real mistake. The check
    therefore fires only on a label that is definitely not CUDA.
"""

from __future__ import annotations

import fastfields.dlpack as _ff

from fastfields.helpers import cuda_backend_message, is_cuda_backend

#: The backend label this namespace was loaded against, for diagnostics.
backend = _ff.backend

if backend is not None and not is_cuda_backend(backend):
    raise ImportError(
        cuda_backend_message(
            "fastfields.cupy",
            backend,
            "every one of its operations runs on cupy arrays in CUDA device "
            "memory and needs the CUDA kernels",
        )
    )
