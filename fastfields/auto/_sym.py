"""Backend-dispatching compact-symmetric ops
([`fastfields.auto`][fastfields.auto]).

Both spellings of each in-place op are dispatched: the trailing-underscore
forms mutate their first argument, which is also the array the backend is
chosen from, so they dispatch as cleanly as the functional ones.

``sym_matvec_backward`` is listed for numpy/cupy only. On torch it is not a
public function at all -- it is the adjoint wired into ``sym_matvec``'s
``autograd.Function``, and the supported way to reach it there is
``sym_matvec(...).backward()``.
"""

from __future__ import annotations

from ._util import make_dispatcher

# unified name -> {backend root: attribute name in that backend package}
_ALIASES = {
    "sym_matvec": {
        "numpy": "sym_matvec",
        "torch": "sym_matvec",
        "cupy": "sym_matvec",
    },
    "sym_matvec_backward": {
        "numpy": "sym_matvec_backward",
        "cupy": "sym_matvec_backward",
    },
    "sym_addmatvec_": {
        "numpy": "sym_addmatvec_",
        "torch": "sym_addmatvec_",
        "cupy": "sym_addmatvec_",
    },
    "sym_submatvec_": {
        "numpy": "sym_submatvec_",
        "torch": "sym_submatvec_",
        "cupy": "sym_submatvec_",
    },
    "sym_solve": {
        "numpy": "sym_solve",
        "torch": "sym_solve",
        "cupy": "sym_solve",
    },
    "sym_solve_": {
        "numpy": "sym_solve_",
        "torch": "sym_solve_",
        "cupy": "sym_solve_",
    },
    "sym_invert": {
        "numpy": "sym_invert",
        "torch": "sym_invert",
        "cupy": "sym_invert",
    },
    "sym_invert_": {
        "numpy": "sym_invert_",
        "torch": "sym_invert_",
        "cupy": "sym_invert_",
    },
}

for _name, _table in _ALIASES.items():
    globals()[_name] = make_dispatcher(_name, _table)

__all__ = list(_ALIASES)
