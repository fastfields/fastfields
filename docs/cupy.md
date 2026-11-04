# fastfields.cupy

**`fastfields.cupy`** runs the fastfields field operators on the **GPU**, straight
on your **CuPy arrays**. Same API as the NumPy package, but the work happens in
device memory and is ordered on CuPy's current CUDA stream.

CuPy itself isn't installed automatically — the right build depends on your CUDA
version — so pull it in with the `cupy` extra (or install a matching
`cupy-cuda12x` yourself).

## Install

```sh
pip install "fastfields[cupy]" \
    --extra-index-url https://fastfields.github.io/whl/cu130/
```

!!! warning "The extra selects CuPy; the index URL selects the CUDA
    kernels"

    These are two separate choices and only the second one can fail
    quietly. `pip install "fastfields[cupy]"` installs CuPy and
    succeeds against a **CPU-only** `fastfields-dlpack`, because
    `pip` has no way to say "any build with CUDA" -- the lane comes
    from the index folder you point it at. `import fastfields.cupy`
    then fails immediately, with a message naming the lane to
    install, because there is no CPU path here to fall back to.
    See [the monorepo note](monorepo.md) for why it works this way.

    No `cu*` lane is published yet. Until one is, this install has
    nothing to resolve to -- the
    [index](https://fastfields.github.io/whl/) lists the lanes that actually exist.

## Use it

```python
import cupy as cp
import fastfields.cupy as ff

mask = cp.zeros((256, 256), "float32")
mask[:, 128] = 1.0

dist = ff.dt_euclidean(mask)      # new array; mask is untouched
ff.dt_euclidean_(mask)            # in-place variant, writes into mask
```

## What's inside

| Operation | Functions |
|---|---|
| **Distance transforms** | `dt_euclidean`, `dt_l1` (along the last axis); point-to-spline `dt_spline_table` / `dt_spline_brent` / `dt_spline_gaussnewton`; point-to-mesh `dt_mesh` |
| **Positive-definite linear algebra** | `sym_matvec`, `sym_addmatvec_`, `sym_submatvec_`, `sym_solve`, `sym_invert` over whole fields of small symmetric matrices |
| **Resampling** | `resample` (spline up/down-sampling), `restriction` (its adjoint), `spline_coeff` (coefficient prefilter) |

Functional wrappers (`dt_euclidean`, `sym_matvec`, …) allocate their outputs and
return new arrays; the trailing-underscore variants (`dt_euclidean_`,
`sym_solve_`, …) write in place. To target a specific stream, wrap the call in a
`with my_stream:` block.

See the [API reference](api/cupy.md) for full signatures and options.
