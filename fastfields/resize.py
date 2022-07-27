__all__ = ['resize', 'restriction', 'restriction_grid',
           'prolongation', 'prolongation_grid']

from .utils import (make_list, movedim,
                    convert_bound, convert_interpolation, convert_align)
import torch
import math
from .ext.bind import (
    resize as _c_resize,
    fmg_prolongation as _c_prolongation,
    fmg_restriction as _c_restriction)


def resize(input, factor=None, bound='dct2', interpolation=1, mode='center',
           shape=None, output=None, adjoint=False, normalize=False):
    """Resize a spatial tensor

    Parameters
    ----------
    input : (N, C, *inshape) tensor
    factor : [sequence of] float, default=1.
    bound : [sequence of] bound_like, default='dct2'
    interpolation : [sequence of] int, default=1
    mode : [sequence of] {'c', 'e', 'f', 'l'}, default='c'
    shape : sequence[int], optional
    output : (N, C, *shape) tensor, optional
    adjoint : bool, default=False
    normalize : bool, default=False

    Returns
    -------
    output : (N, C, *shape) tensor

    """
    bound = convert_bound(make_list(bound), 'enum')
    interpolation = convert_interpolation(make_list(interpolation), 'enum')
    mode = convert_align(make_list(mode), 'enum')
    if output is None:
        if shape:
            output = input.new_zeros([*input.shape[:2], *shape])
        else:
            output = torch.Tensor()
    if torch.is_tensor(factor):
        factor = factor.double().tolist()
    if not factor and not shape:
        raise ValueError('At least one of factor or shape must be provided')
    factor = make_list(factor or [1.])
    return _c_resize(input, output, factor, bound, interpolation,
                     mode, adjoint, normalize)


def prolongation(input, factor=2., bound='dct2', shape=None, output=None):
    """Prolongation of a spatial tensor

    Parameters
    ----------
    input : (N, C, *inshape) tensor
    factor : [sequence of] float, default=2
    bound : [sequence of] bound_like, default='dct2'
    shape : sequence[int], optional
    output : (N, C, *shape) tensor, optional

    Returns
    -------
    output : (N, C, *shape) tensor

    """
    bound = convert_bound(make_list(bound), 'enum')
    if output is None:
        if not shape:
            factor = make_list(factor, 3)
            shape = [int((s*f)//1) for s, f in zip(input.shape[2:], factor)]
        if shape:
            output = input.new_zeros([*input.shape[:2], *shape])
        else:
            output = torch.Tensor()
    return _c_prolongation(input, output, bound)


def restriction(input, factor=2., bound='dct2', shape=None, output=None):
    """Restriction of a spatial tensor

    Parameters
    ----------
    input : (N, C, *inshape) tensor
    factor : [sequence of] float, default=2
    bound : [sequence of] bound_like, default='dct2'
    shape : sequence[int], optional
    output : (N, C, *shape) tensor, optional

    Returns
    -------
    output : (N, C, *shape) tensor

    """
    bound = convert_bound(make_list(bound), 'enum')
    if output is None:
        if not shape:
            factor = make_list(factor, 3)
            shape = [int(math.ceil(s/f)) for s, f in zip(input.shape[2:], factor)]
        if shape:
            output = input.new_zeros([*input.shape[:2], *shape])
        else:
            output = torch.Tensor()
    return _c_restriction(input, output, bound)


def prolongation_grid(input, factor=2., bound='dct2',
                      shape=None, hessian=False, output=None):
    """Prolongation of a spatial tensor

    Parameters
    ----------
    input : (N, *inshape, D) tensor
    factor : [sequence of] float, default=2
    bound : [sequence of] bound_like, default='dct2'
    shape : sequence[int], optional
    output : (N, *shape, D) tensor, optional

    Returns
    -------
    output : (N, *shape, D) tensor

    """
    dim = input.shape[-1]
    input = movedim(input, -1, 1)
    if output is not None:
        output = movedim(output, -1, 1)
    output = prolongation(input, factor, bound, shape, output)
    output = movedim(output, 1, -1)
    factor = [o/i for i, o in zip(input.shape[-dim:], output.shape[-dim:])]
    if not hessian:
        for c in range(dim):
            output[..., c] *= factor[c]
    else:
        count = dim
        for c in range(dim):
            output[..., c] *= factor[c] ** 2
            for cc in range(c+1, dim):
                output[..., count] *= factor[c] * factor[cc]
                count += 1
    return output


def restriction_grid(input, factor=2., bound='dct2',
                     shape=None, hessian=False, output=None):
    """Restriction of a spatial tensor

    Parameters
    ----------
    input : (N, *inshape, D) tensor
    factor : [sequence of] float, default=2
    bound : [sequence of] bound_like, default='dct2'
    shape : sequence[int], optional
    output : (N, *shape, D) tensor, optional

    Returns
    -------
    output : (N, *shape, D) tensor

    """
    dim = input.shape[-1]
    input = movedim(input, -1, 1)
    if output is not None:
        output = movedim(output, -1, 1)
    output = restriction(input, factor, bound, shape, output)
    output = movedim(output, 1, -1)
    factor = [o/i for i, o in zip(input.shape[-dim:], output.shape[-dim:])]
    if not hessian:
        for c in range(dim):
            output[..., c] *= factor[c]
    else:
        count = dim
        for c in range(dim):
            output[..., c] *= factor[c] ** 2
            for cc in range(c+1, dim):
                output[..., count] *= factor[c] * factor[cc]
                count += 1
    return output
