__all__ = ['grid_pull', 'grid_push', 'grid_count', 'grid_grad',
           'identity_grid']

import torch
from .utils import expanded_shape
from .grid_autograd import (GridPull, GridPush, GridCount, GridGrad)


_doc_interpolation = \
    """`interpolation` can be an int, a string or an InterpolationType.
        Possible values are:
            - 0 or 'nearest'
            - 1 or 'linear'
            - 2 or 'quadratic'
            - 3 or 'cubic'
            - 4 or 'fourth'
            - 5 or 'fifth'
            - etc.
        A list of values can be provided, in the order [W, H, D],
        to specify dimension-specific interpolation orders."""

_doc_bound = \
    """`bound` can be an int, a string or a BoundType.
        Possible values are:
            - 'replicate'  or 'nearest'     :  a  a  a  |  a  b  c  d  |  d  d  d
            - 'dct1'       or 'mirror'      :  d  c  b  |  a  b  c  d  |  c  b  a
            - 'dct2'       or 'reflect'     :  c  b  a  |  a  b  c  d  |  d  c  b
            - 'dst1'       or 'antimirror'  : -b -a  0  |  a  b  c  d  |  0 -d -c
            - 'dst2'       or 'antireflect' : -c -b -a  |  a  b  c  d  | -d -c -b
            - 'dft'        or 'wrap'        :  b  c  d  |  a  b  c  d  |  a  b  c
            - 'zero'       or 'zeros'       :  0  0  0  |  a  b  c  d  |  0  0  0
        A list of values can be provided, in the order [W, H, D],
        to specify dimension-specific boundary conditions.
        Note that
        - `dft` corresponds to circular padding
        - `dct2` corresponds to Neumann boundary conditions (symmetric)
        - `dst2` corresponds to Dirichlet boundary conditions (antisymmetric)
        See https://en.wikipedia.org/wiki/Discrete_cosine_transform
            https://en.wikipedia.org/wiki/Discrete_sine_transform"""

_doc_bound_coeff = \
    """`bound` can be an int, a string or a BoundType. 
        Possible values are:
            - 'replicate'  or 'nearest'     :  a  a  a  |  a  b  c  d  |  d  d  d
            - 'dct1'       or 'mirror'      :  d  c  b  |  a  b  c  d  |  c  b  a
            - 'dct2'       or 'reflect'     :  c  b  a  |  a  b  c  d  |  d  c  b
            - 'dst1'       or 'antimirror'  : -b -a  0  |  a  b  c  d  |  0 -d -c
            - 'dst2'       or 'antireflect' : -c -b -a  |  a  b  c  d  | -d -c -b
            - 'dft'        or 'wrap'        :  b  c  d  |  a  b  c  d  |  a  b  c
            - 'zero'       or 'zeros'       :  0  0  0  |  a  b  c  d  |  0  0  0
        A list of values can be provided, in the order [W, H, D],
        to specify dimension-specific boundary conditions.
        Note that
        - `dft` corresponds to circular padding
        - `dct1` corresponds to mirroring about the center of the first/last voxel
        - `dct2` corresponds to mirroring about the edge of the first/last voxel
        See https://en.wikipedia.org/wiki/Discrete_cosine_transform
            https://en.wikipedia.org/wiki/Discrete_sine_transform
    
        /!\ Only 'dct1', 'dct2' and 'dft' are implemented for interpolation
            orders >= 6."""

_ref_coeff = \
    """..[1]  M. Unser, A. Aldroubi and M. Eden.
           "B-Spline Signal Processing: Part I-Theory,"
           IEEE Transactions on Signal Processing 41(2):821-832 (1993).
    ..[2]  M. Unser, A. Aldroubi and M. Eden.
           "B-Spline Signal Processing: Part II-Efficient Design and Applications,"
           IEEE Transactions on Signal Processing 41(2):834-848 (1993).
    ..[3]  M. Unser.
           "Splines: A Perfect Fit for Signal and Image Processing,"
           IEEE Signal Processing Magazine 16(6):22-38 (1999).
    """


def _preproc(grid, input=None, mode=None):
    """Preprocess tensors for pull/push/count/grad

    Low level bindings expect inputs of shape
    [batch, channel, *spatial] and [batch, *spatial, dim], whereas
    the high level python API accepts inputs of shape
    [..., [channel], *spatial] and [..., *spatial, dim].

    This function broadcasts and reshapes the input tensors accordingly.
            /!\\ This *can* trigger large allocations /!\\
    """
    dim = grid.shape[-1]
    if input is None:
        spatial = grid.shape[-dim - 1:-1]
        batch = grid.shape[:-dim - 1]
        grid = grid.reshape([-1, *spatial, dim])
        info = dict(batch=batch, channel=[1] if batch else [], dim=dim)
        return grid, info

    grid_spatial = grid.shape[-dim - 1:-1]
    grid_batch = grid.shape[:-dim - 1]
    input_spatial = input.shape[-dim:]
    channel = 0 if input.dim() == dim else input.shape[-dim - 1]
    input_batch = input.shape[:-dim - 1]

    if mode == 'push':
        grid_spatial = input_spatial = expanded_shape(grid_spatial, input_spatial)

    # broadcast and reshape
    batch = expanded_shape(grid_batch, input_batch)
    grid = grid.expand([*batch, *grid_spatial, dim])
    grid = grid.reshape([-1, *grid_spatial, dim])
    input = input.expand([*batch, channel or 1, *input_spatial])
    input = input.reshape([-1, channel or 1, *input_spatial])

    out_channel = [channel] if channel else ([1] if batch else [])
    info = dict(batch=batch, channel=out_channel, dim=dim)
    return grid, input, info


def _postproc(out, shape_info, mode):
    """Postprocess tensors for pull/push/count/grad"""
    dim = shape_info['dim']
    if mode != 'grad':
        spatial = out.shape[-dim:]
        feat = []
    else:
        spatial = out.shape[-dim - 1:-1]
        feat = [out.shape[-1]]
    batch = shape_info['batch']
    channel = shape_info['channel']

    out = out.reshape([*batch, *channel, *spatial, *feat])
    return out


def grid_pull(input, grid, interpolation='linear', bound='zero',
              extrapolate=False):
    """Sample an image with respect to a deformation field.

    Notes
    -----
    {interpolation}

    {bound}

    If the input dtype is not a floating point type, the input image is
    assumed to contain labels. Then, unique labels are extracted
    and resampled individually, making them soft labels. Finally,
    the label map is reconstructed from the individual soft labels by
    assigning the label with maximum soft value.

    Parameters
    ----------
    input : (..., [channel], *inshape) tensor
        Input image.
    grid : (..., *outshape, dim) tensor
        Transformation field.
    interpolation : int or sequence[int], default=1
        Interpolation order.
    bound : BoundType or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool or int, default=True
        Extrapolate out-of-bound data.

    Returns
    -------
    output : (..., [channel], *outshape) tensor
        Deformed image.

    """
    grid, input, shape_info = _preproc(grid, input)
    batch, channel = input.shape[:2]
    dim = grid.shape[-1]

    if not input.dtype.is_floating_point:
        # label map -> specific processing
        out = input.new_zeros([batch, channel, *grid.shape[1:-1]])
        pmax = grid.new_zeros([batch, channel, *grid.shape[1:-1]])
        for label in input.unique():
            soft = (input == label).to(grid.dtype)
            soft = GridPull.apply(soft, grid, interpolation, bound, extrapolate)
            out[soft > pmax] = label
            pmax = torch.max(pmax, soft)
    else:
        out = GridPull.apply(input, grid, interpolation, bound, extrapolate)

    return _postproc(out, shape_info, mode='pull')


def grid_push(input, grid, shape=None, interpolation='linear', bound='zero',
              extrapolate=False):
    """Splat an image with respect to a deformation field (pull adjoint).

    Notes
    -----
    {interpolation}

    {bound}

    Parameters
    ----------
    input : (..., [channel], *inshape) tensor
        Input image.
    grid : (..., *inshape, dim) tensor
        Transformation field.
    shape : sequence[int], default=inshape
        Output shape
    interpolation : int or sequence[int], default=1
        Interpolation order.
    bound : BoundType, or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool or int, default=True
        Extrapolate out-of-bound data.

    Returns
    -------
    output : (..., [channel], *shape) tensor
        Spatted image.

    """
    grid, input, shape_info = _preproc(grid, input, mode='push')
    dim = grid.shape[-1]

    if shape is None:
        shape = tuple(input.shape[2:])

    out = GridPush.apply(input, grid, shape, interpolation, bound, extrapolate)
    return _postproc(out, shape_info, mode='push')


def grid_count(grid, shape=None, interpolation='linear', bound='zero',
               extrapolate=False):
    """Splatting weights with respect to a deformation field (pull adjoint).

    Notes
    -----
    {interpolation}

    {bound}

    Parameters
    ----------
    grid : (..., *inshape, dim) tensor
        Transformation field.
    shape : sequence[int], default=inshape
        Output shape
    interpolation : int or sequence[int], default=1
        Interpolation order.
    bound : BoundType, or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool or int, default=True
        Extrapolate out-of-bound data.

    Returns
    -------
    output : (..., [1], *shape) tensor
        Splatted weights.

    """
    grid, shape_info = _preproc(grid)
    out = GridCount.apply(grid, shape, interpolation, bound, extrapolate)
    return _postproc(out, shape_info, mode='count')


def grid_grad(input, grid, interpolation='linear', bound='zero',
              extrapolate=False):
    """Sample spatial gradients of an image with respect to a deformation field.

    Notes
    -----
    {interpolation}

    {bound}

    Parameters
    ----------
    input : (..., [channel], *inshape) tensor
        Input image.
    grid : (..., *inshape, dim) tensor
        Transformation field.
    shape : sequence[int], default=inshape
        Output shape
    interpolation : int or sequence[int], default=1
        Interpolation order.
    bound : BoundType, or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool or int, default=True
        Extrapolate out-of-bound data.

    Returns
    -------
    output : (..., [channel], *shape, dim) tensor
        Sampled gradients.

    """
    grid, input, shape_info = _preproc(grid, input)
    out = GridGrad.apply(input, grid, interpolation, bound, extrapolate)
    return _postproc(out, shape_info, mode='grad')


grid_pull.__doc__ = grid_pull.__doc__.format(
    interpolation=_doc_interpolation, bound=_doc_bound)
grid_push.__doc__ = grid_push.__doc__.format(
    interpolation=_doc_interpolation, bound=_doc_bound)
grid_count.__doc__ = grid_count.__doc__.format(
    interpolation=_doc_interpolation, bound=_doc_bound)
grid_grad.__doc__ = grid_grad.__doc__.format(
    interpolation=_doc_interpolation, bound=_doc_bound)


def identity_grid(shape, dtype=None, device=None):
    """Returns an identity deformation field.

    Parameters
    ----------
    shape : (dim,) sequence of int
        Spatial dimension of the field.
    dtype : torch.dtype, default=`get_default_dtype()`
        Data type.
    device torch.device, optional
        Device.

    Returns
    -------
    grid : (*shape, dim) tensor
        Transformation field

    """
    mesh1d = [torch.arange(float(s), dtype=dtype, device=device)
              for s in shape]
    grid = torch.meshgrid(*mesh1d)
    grid = torch.stack(grid, dim=-1)
    return grid
