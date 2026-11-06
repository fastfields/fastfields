"""Shared enums and argument normalisers for the fastfields namespaces.

`fastfields.numpy`, `fastfields.torch`, `fastfields.cupy` and
`fastfields.auto` all need the same logic that does not depend on an array
library:

- the [`Spline`][fastfields.helpers.Spline] and
  [`Bound`][fastfields.helpers.Bound] enums and the normalisers that turn an
  integer, an enum member or a name into one of them;
- the resolution of `factor`, `shape` and `anchor` into the per-axis `scale`
  and scalar `shift` that the resampling kernels consume;
- the checks that decide whether the installed `fastfields-dlpack` is a CUDA
  build.

It lives here, in one place, so that each of them imports it instead of
carrying a copy.

The package is pure Python and imports nothing but the standard library: no
NumPy, PyTorch or CuPy, and not the compiled `fastfields.dlpack` bindings. That
is the reason it exists apart from them. A fix to this logic ships as an
ordinary pure-Python release, without recompiling the C++/CUDA chain or
rebuilding the per-platform wheels. It is a sibling of `fastfields.dlpack`,
not a layer over it, and neither depends on the other.
"""

from __future__ import annotations

import re as _re
from enum import IntEnum
from typing import Optional, Sequence

__all__ = [
    "__version__",
    "Spline",
    "Bound",
    "as_spline",
    "as_bound",
    "normalize_shape",
    "infer_ndim",
    "check_ndim",
    "resolve_out_spatial",
    "anchor_scale_shift",
    "is_cuda_backend",
    "cuda_backend_message",
]

# -- version ---------------------------------------------------------------
# ``_version.py`` is generated at build time by versioningit; a bare source
# tree that was never built has no such file.
#
# This is the whole `fastfields` distribution's version, not just this
# subpackage's: the five merged subpackages used to be five distributions with
# five separate `_version.py` files, and versioningit writes exactly one. It
# lives here because `fastfields.helpers` is the bottom of this distribution's
# internal stack -- the other four import it and it imports none of them, so
# reading the version can never pull in an array library, and `auto` (which
# imports this module) could not have held it without a cycle.
try:
    from ._version import __version__
except ImportError:  # pragma: no cover - source tree without a build
    __version__ = "0+unknown"


# -- compute-backend requirement -------------------------------------------
# ``fastfields-dlpack`` ships the compiled kernels and records which compute
# backend it was built for in its version's PEP 440 *local* segment,
# PyTorch-style: ``0.0.dev0+cpu``, ``+cu118``, ``+cu126``, ``+cu130``. It
# exposes the leading component as ``fastfields.dlpack.backend``.
#
# Which lane a user gets is decided by the wheel-index folder they install
# from, not by the dependency graph -- pip cannot express "any build with
# CUDA" -- so no package in this distribution pins a lane, and the namespaces
# that need one enforce it at runtime instead. The two predicates below are
# the shared half of that: they are pure functions of the label, and
# deliberately do NOT import ``fastfields.dlpack`` themselves, which is what
# keeps this module stdlib-only. Each caller reads
# ``fastfields.dlpack.backend`` itself and decides when to look (at import,
# for fastfields.cupy; only on meeting a CUDA tensor, for fastfields.torch).

#: Base URL of the fastfields wheel index; one folder per compute backend.
WHEEL_INDEX_URL = "https://fastfields.github.io/whl"

_CUDA_BACKEND_RE = _re.compile(r"\Acu[0-9]+\Z")


def is_cuda_backend(backend: Optional[str]) -> bool:
    """Tell whether a backend label names a CUDA build of `fastfields-dlpack`.

    The label is what `fastfields.dlpack.backend` reports: `"cpu"`, `"cu126"`
    and so on, or `None` for a build that carries no label, such as an
    unversioned source checkout.

    `None` means unknown, not CPU. A caller that rejects non-CUDA builds should
    therefore test `backend is not None and not is_cuda_backend(backend)`
    instead of negating this function: a source build cannot be verified, and
    refusing it would block GPU development from a checkout for no gain in
    safety.

    Parameters
    ----------
    backend : str or None
        A compute-backend label.

    Returns
    -------
    bool
        True for `cu` followed by digits, False for anything else, `None`
        included.
    """
    return bool(backend) and bool(_CUDA_BACKEND_RE.match(backend))


def cuda_backend_message(
    namespace: str,
    backend: Optional[str],
    because: str,
) -> str:
    """Build the error message for a `fastfields-dlpack` build without CUDA.

    [`fastfields.cupy`][fastfields.cupy] raises it at import, and
    [`fastfields.torch`][fastfields.torch] when a CUDA tensor meets a CPU-only
    build. Sharing the function gives both the same wording, which names the
    index URL and the `pip` command that fixes the problem.

    Parameters
    ----------
    namespace : str
        The namespace that is refusing, for example `"fastfields.cupy"`.
    backend : str or None
        The installed `fastfields.dlpack.backend`.
    because : str
        A clause that completes "cannot work because ...", saying what in the
        caller's situation requires CUDA kernels.

    Returns
    -------
    str
        The complete, multi-line message.
    """
    installed = (
        f"a {backend!r} build" if backend else "a build with no backend label"
    )
    return (
        f"{namespace} cannot work because {because}, but the installed "
        f"fastfields-dlpack is {installed}. The compute backend is chosen by "
        f"the wheel-index folder you install from, not by the dependency "
        f"graph, so installing a CUDA build means pointing pip at a CUDA "
        f"lane:\n"
        f"\n"
        f"    pip install --force-reinstall fastfields-dlpack \\\n"
        f"        --extra-index-url {WHEEL_INDEX_URL}/cu126/\n"
        f"\n"
        f"Lanes live at {WHEEL_INDEX_URL}/ (cpu, cu118, cu126, cu130); pick "
        f"the one matching your driver. Note that a CUDA lane may not be "
        f"published yet -- the index lists the lanes that are actually "
        f"available, and installing from an unpublished one resolves to "
        f"nothing."
    )


class Spline(IntEnum):
    """Order of the spline used for interpolation.

    The wrappers accept it as the `order` argument, as an integer, as a member
    of this enum or as a name such as `"cubic"`; see
    [`as_spline`][fastfields.helpers.as_spline]. The raw `fastfields.dlpack`
    bindings and the C++ library call the same argument `spline`.
    """

    Nearest = 0
    Linear = 1
    Quadratic = 2
    Cubic = 3
    FourthOrder = 4
    FifthOrder = 5
    SixthOrder = 6
    SeventhOrder = 7


class Bound(IntEnum):
    """Boundary condition, applied to every spatial axis.

    The wrappers accept it as the `bound` argument, as an integer, as a member
    of this enum or as a name such as `"dct2"`; see
    [`as_bound`][fastfields.helpers.as_bound]. The comment beside each member
    says how the signal is extended beyond the last voxel.
    """

    Zero = 0  # zero outside the FOV
    Replicate = 1  # clip coordinates
    DCT1 = 2  # symmetric w.r.t. voxel centre
    DCT2 = 3  # symmetric w.r.t. voxel edge (Neumann)
    DST1 = 4  # antisymmetric w.r.t. voxel centre
    DST2 = 5  # antisymmetric w.r.t. voxel edge (Dirichlet)
    DFT = 6  # circular / wrap around
    NoCheck = 7  # assume coordinates are inbound


# --------------------------------------------------------------------------- #
# order / bound normalisation                                                 #
# --------------------------------------------------------------------------- #

_SPLINE_ALIASES = {
    "nearest": Spline.Nearest,
    "constant": Spline.Nearest,
    "linear": Spline.Linear,
    "quadratic": Spline.Quadratic,
    "cubic": Spline.Cubic,
    "fourth": Spline.FourthOrder,
    "fifth": Spline.FifthOrder,
    "sixth": Spline.SixthOrder,
    "seventh": Spline.SeventhOrder,
}

_BOUND_ALIASES = {
    "zero": Bound.Zero,
    "zeros": Bound.Zero,
    "replicate": Bound.Replicate,
    "nearest": Bound.Replicate,
    "dct1": Bound.DCT1,
    "dct2": Bound.DCT2,
    "neumann": Bound.DCT2,
    "reflect": Bound.DCT2,
    "dst1": Bound.DST1,
    "dst2": Bound.DST2,
    "dirichlet": Bound.DST2,
    "dft": Bound.DFT,
    "wrap": Bound.DFT,
    "circular": Bound.DFT,
    "nocheck": Bound.NoCheck,
}


def as_spline(value: int | str | Spline) -> int:
    """Turn a spline-order argument into an integer from 0 to 7.

    Parameters
    ----------
    value : int or str or Spline
        An integer, a [`Spline`][fastfields.helpers.Spline] member, or one of
        the names `"nearest"` (or `"constant"`), `"linear"`, `"quadratic"`,
        `"cubic"`, `"fourth"`, `"fifth"`, `"sixth"` and `"seventh"`, in any
        case.

    Returns
    -------
    int
        The spline order.

    Raises
    ------
    ValueError
        If the name is unknown or the integer is outside 0 to 7.
    """
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _SPLINE_ALIASES:
            raise ValueError(
                f"unknown spline order {value!r}; "
                f"expected an int 0..7 or one of {sorted(_SPLINE_ALIASES)}"
            )
        return int(_SPLINE_ALIASES[key])
    ivalue = int(value)
    if not 0 <= ivalue <= 7:
        raise ValueError(f"spline order must be in 0..7, got {ivalue}")
    return ivalue


def as_bound(value: int | str | Bound) -> int:
    """Turn a boundary-condition argument into an integer from 0 to 7.

    Parameters
    ----------
    value : int or str or Bound
        An integer, a [`Bound`][fastfields.helpers.Bound] member, or a name, in
        any case. The names are `"zero"` (or `"zeros"`), `"replicate"` (or
        `"nearest"`), `"dct1"`, `"dct2"` (or `"neumann"`, `"reflect"`),
        `"dst1"`, `"dst2"` (or `"dirichlet"`), `"dft"` (or `"wrap"`,
        `"circular"`) and `"nocheck"`.

    Returns
    -------
    int
        The boundary condition as an integer.

    Raises
    ------
    ValueError
        If the name is unknown or the integer is outside 0 to 7.
    """
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _BOUND_ALIASES:
            raise ValueError(
                f"unknown boundary condition {value!r}; "
                f"expected an int 0..7 or one of {sorted(_BOUND_ALIASES)}"
            )
        return int(_BOUND_ALIASES[key])
    ivalue = int(value)
    if not 0 <= ivalue <= 7:
        raise ValueError(f"boundary condition must be in 0..7, got {ivalue}")
    return ivalue


# --------------------------------------------------------------------------- #
# resize shape / ndim resolution                                              #
# --------------------------------------------------------------------------- #


def normalize_shape(shape: int | Sequence[int], ndim: int) -> list[int]:
    """Expand a shape argument to a list of `ndim` integers.

    Parameters
    ----------
    shape : int or sequence of int
        A single size, used for every axis, or one size per axis.
    ndim : int
        Number of axes.

    Returns
    -------
    list of int
        The sizes, one per axis.

    Raises
    ------
    ValueError
        If a sequence is given and its length is not `ndim`.
    """
    if isinstance(shape, int):
        shape = [shape] * ndim
    shape = list(shape)
    if len(shape) != ndim:
        raise ValueError(f"Expected shape of length ndim={ndim}, got {shape}.")
    return [int(s) for s in shape]


def infer_ndim(
    ndim: Optional[int],
    factor: float | Sequence[float] | None,
    shape: int | Sequence[int] | None,
) -> int:
    """Work out how many trailing axes a resampling call applies to.

    The rule is, in order: an explicit `ndim` wins; otherwise the length of
    `shape` or `factor`, if one of them is a sequence; otherwise 1.

    Parameters
    ----------
    ndim : int or None
        The explicit number of axes, if given.
    factor : float or sequence of float or None
        The size multiplier.
    shape : int or sequence of int or None
        The output size.

    Returns
    -------
    int
        The number of trailing axes.
    """
    if ndim is not None:
        return int(ndim)
    if shape is not None and not isinstance(shape, int):
        return len(list(shape))
    if factor is not None and not isinstance(factor, (int, float)):
        return len(list(factor))
    return 1


def check_ndim(ndim: int, arr_ndim: int) -> None:
    """Check that `ndim` is between 1 and the number of axes of the array.

    Parameters
    ----------
    ndim : int
        Number of trailing axes to operate on.
    arr_ndim : int
        Number of axes of the array.

    Raises
    ------
    ValueError
        If `ndim` is outside `1..arr_ndim`.
    """
    if ndim < 1 or ndim > arr_ndim:
        raise ValueError(f"ndim must be in 1..{arr_ndim}, got {ndim}")


def resolve_out_spatial(
    spatial_in: Sequence[int],
    ndim: int,
    factor: float | Sequence[float] | None,
    shape: int | Sequence[int] | None,
) -> tuple[int, ...]:
    """Work out the output size of a resampling call.

    Give `shape` for an explicit size, or `factor` to scale the input size;
    they are mutually exclusive, and with neither the size is unchanged. A
    scalar `factor` applies to every axis, and each size becomes `max(1,
    round(n * factor))`.

    Parameters
    ----------
    spatial_in : sequence of int
        Input sizes of the resampled axes.
    ndim : int
        Number of resampled axes.
    factor : float or sequence of float or None
        The size multiplier.
    shape : int or sequence of int or None
        The output size. When given it takes precedence over `factor`.

    Returns
    -------
    tuple of int
        The output sizes.

    Raises
    ------
    ValueError
        If `shape` or `factor` is a sequence whose length is not `ndim`.
    """
    if shape is not None:
        return tuple(normalize_shape(shape, ndim))
    if factor is not None:
        if isinstance(factor, (int, float)):
            factors = [float(factor)] * ndim
        else:
            factors = [float(f) for f in factor]
        if len(factors) != ndim:
            raise ValueError(
                f"Expected factor of length ndim={ndim}, got {factors}."
            )
        return tuple(
            max(1, int(round(n * f))) for n, f in zip(spatial_in, factors)
        )
    return tuple(int(n) for n in spatial_in)  # identity


# --------------------------------------------------------------------------- #
# torch-interpol anchor conventions                                           #
# --------------------------------------------------------------------------- #

# Each anchor is identified by its first (lower-cased) letter, so both the full
# name ("centers") and the abbreviation ("c") are accepted, mirroring
# ``interpol.resize``.
_ANCHORS = ("c", "e", "f", "l")
_ANCHOR_SHIFT = {"e": 0.5, "f": 0.0, "l": 1.0}


def anchor_scale_shift(
    anchor: str,
    inshape: Sequence[int],
    outshape: Sequence[int],
    ndim: int,
) -> tuple[list[float], float]:
    """Turn an `anchor` into the `scale` and `shift` of the resampling kernel.

    For output index `loc`, the kernel samples the input at
    `scale[d] * loc + shift * (scale[d] - 1)` along axis `d`. The four anchors
    choose which points of the two grids line up:

    | `anchor`    | `scale[d]`          | `shift` |
    |-------------|---------------------|---------|
    | `"centers"` | `(in-1) / (out-1)`  | `0.0`   |
    | `"edges"`   | `in / out`          | `0.5`   |
    | `"first"`   | `in / out`          | `0.0`   |
    | `"last"`    | `in / out`          | `1.0`   |

    With `"centers"` the first and last samples coincide; the scale is 1 along
    an axis where either size is 1. With `"edges"` the outer edges of the
    voxels coincide, with `"first"` the first voxel is the anchor and with
    `"last"` the last one. The names follow `interpol.resize`.

    Parameters
    ----------
    anchor : str
        `"centers"`, `"edges"`, `"first"` or `"last"`, or their initial
        letters. Only the first letter is looked at, in any case.
    inshape, outshape : sequence of int
        Input and output sizes of the resampled axes, `ndim` values each.
    ndim : int
        Number of resampled axes.

    Returns
    -------
    scale : list of float
        Input-index step per output-index step, one value per axis.
    shift : float
        The shift, shared by all axes.

    Raises
    ------
    ValueError
        If `anchor` does not start with `c`, `e`, `f` or `l`.
    """
    key = str(anchor)[:1].lower()
    if key not in _ANCHORS:
        raise ValueError(
            f"anchor must be one of centers/edges/first/last, got {anchor!r}"
        )
    if key == "c":
        scale = [
            ((inshape[d] - 1) / (outshape[d] - 1))
            if (inshape[d] > 1 and outshape[d] > 1)
            else 1.0
            for d in range(ndim)
        ]
        return scale, 0.0
    scale = [float(inshape[d]) / float(outshape[d]) for d in range(ndim)]
    return scale, _ANCHOR_SHIFT[key]
