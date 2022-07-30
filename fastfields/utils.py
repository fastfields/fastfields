__all__ = ['BoundType', 'InterpolationType', 'GridAlignType',
           'convert_bound', 'convert_interpolation', 'convert_align',
           'custom_fwd', 'custom_bwd', 'make_list', 'vector_to_list',
           'expanded_shape', 'movedim']

import torch
from .ext.bind import InterpolationType, BoundType, GridAlignType

try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except ImportError:
    custom_fwd = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda x: x)
    custom_bwd = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda x: x)


def make_list(x, n=None):
    if not isinstance(x, (list, tuple)):
        x = [x]
    x = list(x)
    if n and x:
        x = x + max(0, n - len(x)) * x[-1:]
    return x


def vector_to_list(x, dtype=None, n=None):
    if torch.is_tensor(x):
        x = x.double().tolist()
    x = make_list(x, n)
    if dtype:
        x = list(map(dtype, x))
    return x


def expanded_shape(*shapes, side='left'):
    """Expand input shapes according to broadcasting rules

    Parameters
    ----------
    *shapes : sequence[int]
        Input shapes
    side : {'left', 'right'}, default='left'
        Side to add singleton dimensions.

    Returns
    -------
    shape : tuple[int]
        Output shape

    Raises
    ------
    ValueError
        If shapes are not compatible for broadcast.

    """
    def error(s0, s1):
        raise ValueError('Incompatible shapes for broadcasting: {} and {}.'
                         .format(s0, s1))

    # 1. nb dimensions
    nb_dim = 0
    for shape in shapes:
        nb_dim = max(nb_dim, len(shape))

    # 2. enumerate
    shape = [1] * nb_dim
    for i, shape1 in enumerate(shapes):
        pad_size = nb_dim - len(shape1)
        ones = [1] * pad_size
        if side == 'left':
            shape1 = [*ones, *shape1]
        else:
            shape1 = [*shape1, *ones]
        shape = [max(s0, s1) if s0 == 1 or s1 == 1 or s0 == s1
                 else error(s0, s1) for s0, s1 in zip(shape, shape1)]

    return tuple(shape)


if hasattr(torch, 'movedim'):
    movedim = torch.movedim
else:
    def movedim(input, source, destination):
        """Move the position of exactly one dimension"""
        dim = input.dim()

        source = dim + source if source < 0 else source
        destination = dim + destination if destination < 0 else destination
        permutation = list(range(dim))
        del permutation[source]
        permutation.insert(destination, source)
        return input.permute(*permutation)


def convert_bound(bound, as_type='str'):
    """Convert boundary type to FastFields's convention.

    Parameters
    ----------
    bound : [list of] str or bound_like
        Boundary condition in any convention
    as_type : {'str', 'enum', 'int'}, default='str'
        Return BoundType or int rather than str

    Returns
    -------
    bound : [list of] str or BoundType
        Boundary condition in FastFields's convention

    """
    intype = type(bound)
    if not isinstance(bound, (list, tuple)):
        bound = [bound]
    obound = []
    for b in bound:
        b = b.lower() if isinstance(b, str) else b
        if b in ('replicate', 'repeat', 'border', 'nearest', BoundType.replicate):
            obound.append('replicate')
        elif b in ('zero', 'zeros', 'constant', BoundType.zero):
            obound.append('zero')
        elif b in ('dct2', 'reflect', 'reflection', 'neumann', BoundType.dct2):
            obound.append('dct2')
        elif b in ('dct1', 'mirror', BoundType.dct1):
            obound.append('dct1')
        elif b in ('dft', 'wrap', 'circular', BoundType.dft):
            obound.append('dft')
        elif b in ('dst2', 'antireflect', 'dirichlet', BoundType.dst2):
            obound.append('dst2')
        elif b in ('dst1', 'antimirror', BoundType.dst1):
            obound.append('dst1')
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if as_type in ('enum', 'int', int):
        obound = list(map(lambda b: getattr(BoundType, b), obound))
        if as_type in ('int', int):
            raise NotImplementedError
    if issubclass(intype, (list, tuple)):
        obound = intype(obound)
    else:
        obound = obound[0]
    return obound


def convert_interpolation(inter, as_type='str'):
    """Convert interpolation order to FastFields's convention.

    Parameters
    ----------
    inter : [sequence of] int or str or InterpolationType
    as_type : {'str', 'enum', 'int'}, default='int'

    Returns
    -------
    inter : [sequence of] int or InterpolationType

    """
    intype = type(inter)
    if not isinstance(inter, (list, tuple)):
        inter = [inter]
    ointer = []
    for o in inter:
        o = o.lower() if isinstance(o, str) else o
        if o in (0, 'nearest', InterpolationType.nearest):
            ointer.append(0)
        elif o in (1, 'linear', InterpolationType.linear):
            ointer.append(1)
        elif o in (2, 'quadratic', InterpolationType.quadratic):
            ointer.append(2)
        elif o in (3, 'cubic', InterpolationType.cubic):
            ointer.append(3)
        elif o in (4, 'fourth', InterpolationType.fourth):
            ointer.append(4)
        elif o in (5, 'fifth', InterpolationType.fifth):
            ointer.append(5)
        elif o in (6, 'sixth', InterpolationType.sixth):
            ointer.append(6)
        elif o in (7, 'seventh', InterpolationType.seventh):
            ointer.append(7)
        else:
            raise ValueError(f'Unknown interpolation order {o}')
    if as_type in ('enum', 'str', str):
        ointer = list(map(InterpolationType, ointer))
        if as_type in ('str', str):
            ointer = [o.name for o in ointer]
    if issubclass(intype, (list, tuple)):
        ointer = intype(ointer)
    else:
        ointer = ointer[0]
    return ointer


def convert_align(align, as_type='str'):
    """Convert alignment mode to FastFields's convention.

    Parameters
    ----------
    align : [list of] str or align_like
        Alignment mode in any convention
    as_type : {'str', 'enum', 'int'}, default='str'
        Return GridAlignType or int rather than str

    Returns
    -------
    align : [list of] str or GridAlignType
        Alignment mode in FastFields's convention

    """

    intype = type(align)
    align = make_list(align)
    oalign = []
    for b in align:
        b = b.lower() if isinstance(b, str) else b
        if b[0] == 'c' or b == GridAlignType.center:
            oalign.append('center')
        elif b[0] == 'e' or b == GridAlignType.edge:
            oalign.append('edge')
        elif b[0] == 'f' or b == GridAlignType.first:
            oalign.append('first')
        elif b[0] == 'l' or b == GridAlignType.last:
            oalign.append('last')
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if as_type in ('enum', 'int', int):
        oalign = list(map(lambda b: getattr(GridAlignType, b), oalign))
    if issubclass(intype, (list, tuple)):
        oalign = intype(oalign)
    else:
        oalign = oalign[0]
    return oalign
