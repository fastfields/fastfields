# fastfields

`fastfields` is a PyTorch extension written in C++/CUDA specialized for 
dense scalar or vector fields. This package contains a limited set of tools
that are implemented in C++/CUDA, and is aimed to serve as a core dependency of 
higher level packages (such as [`nitorch`](https://github.com/balbasty/nitorch)).


## Installation

/!\ Currently, `fastfields` is only available on Linux and MacOS.

PyTorch extensions are linked against `libtorch`, the C++/CUDA library on which 
Pytorch is built. Because the ABI (Application Binary Interface) of `libtorch` 
is not stable, extensions depend strictly on the version of `libtorch` that was
used at compilation time. We therefore had to compile each version of 
`fastfields` against all possible versions (and CUDA or CPU backend) of 
`libtorch`. This makes distribution less straightforward than for pure Python 
packages and means that we cannot use the PyPi index. We instead uploaded all
compiled versions of `fastfields` on 
[GemFury](https://pypi.fury.io/balbasty/fastfields).

To ensure compatibility between `fastfields` and `torch`, we advise installing 
them together with `pip`:
```shell
pip install torch==1.7.1+cu101 fastfields==0.1.0+torch1.7.1cu101 \
    -f https://download.pytorch.org/whl/torch_stable.html \
    -f https://pypi.fury.io/balbasty/fastfields
```


## Sampling spline-encoded fields (and its adjoint)

A first set of tools is concerned with N-dimensional high-order spline 
interpolation. A version of these tools written in TorchScript is available 
in the [`torch-interpol`](https://github.com/balbasty/torch-interpol) package.
Note that **all functions in this section are automatically differentiable**.

We implement an "interpolation" function akin to PyTorch's 
[`grid_sample`](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html),
except that is supports spline encoding up to order 7, more out-of-bound 
extrapolation methods, and does not encode coordinates in (-1, ..., 1) but in 
(0, ..., N-1).

`interpolation` can be an `int` or a `str`.
A list of values can be provided to specify dimension-specific interpolation orders.
Possible values are:
- 0 or 'nearest'
- 1 or 'linear'
- 2 or 'quadratic'
- 3 or 'cubic'
- 4 or 'fourth'
- 5 or 'fifth'
- etc.

`bound` must be a `str`. 
A list of values can be provided to specify dimension-specific boundary conditions.
Possible values are:
```
- 'replicate'  or 'nearest'     :  a  a  a  |  a  b  c  d  |  d  d  d
- 'dct1'       or 'mirror'      :  d  c  b  |  a  b  c  d  |  c  b  a
- 'dct2'       or 'reflect'     :  c  b  a  |  a  b  c  d  |  d  c  b
- 'dst1'       or 'antimirror'  : -b -a  0  |  a  b  c  d  |  0 -d -c
- 'dst2'       or 'antireflect' : -c -b -a  |  a  b  c  d  | -d -c -b
- 'dft'        or 'wrap'        :  b  c  d  |  a  b  c  d  |  a  b  c
- 'zero'       or 'zeros'       :  0  0  0  |  a  b  c  d  |  0  0  0
```

Note that
- `dft` corresponds to the boundary conditions of a [DFT](https://en.wikipedia.org/wiki/Discrete_Fourier_transform) (circular)
- `dct2` corresponds to the boundary conditions of a [DCT-II](https://en.wikipedia.org/wiki/Discrete_cosine_transform), or to Neumann boundary conditions (symmetric)
- `dst2` corresponds to the boundary conditions of a [DCT-II](https://en.wikipedia.org/wiki/Discrete_sine_transform), or to Dirichlet boundary conditions (antisymmetric)

```python
grid_pull(input, grid, interpolation='linear', bound='zero', extrapolate=False)
"""Sample an image with respect to a deformation field.

Note
----
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
bound : str or sequence[str], default='zero'
    Boundary conditions.
extrapolate : bool or int, default=True
    Extrapolate out-of-bound data.

Returns
-------
output : (..., [channel], *outshape) tensor
    Deformed image.
"""
```

Note that this function does not _interpolate_ the input tensor, but assumes 
instead that it contains spline coefficients that encode a continuous 
function. To perform interpolation, a dense field must first be converted into 
interpolating spline coefficients 
(see [Unser et al. (1993)](http://bigwww.epfl.ch/publications/unser9301.html)), 
which can be efficiently done using a separable recursive filter. Such a 
_prefiltering_ function is implemented (using TorchScript) in 
[`torch-interpol`](https://github.com/balbasty/torch-interpol).

Since splines are differentiable, we can use the same encoding to sample 
directional derivatives of the input tensor:

```python
grid_grad(input, grid, interpolation='linear', bound='zero', extrapolate=False)
"""Sample spatial gradients of an image with respect to a deformation field.

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
bound : str, or sequence[str], default='zero'
    Boundary conditions.
extrapolate : bool or int, default=True
    Extrapolate out-of-bound data.

Returns
-------
output : (..., [channel], *shape, dim) tensor
    Sampled gradients.

"""
```

The function implemented by `grid_pull` is linear with respect to the input 
tensor and can be thought of as the matrix-vector product **Bv**, where
**B** is a matrix that encodes the sampling grid (`grid`) and spline 
coefficients, while **v** is a vector that contains the flattened `input`. The 
adjoint operation **B**<sup>T</sup>**v**, which utilizes the transposed matrix 
**B**<sup>T</sup>, is also linear in **v**. We implement it efficiently as the 
function `grid_push`:

```python
grid_push(input, grid, shape=None, 
          interpolation='linear', bound='zero', extrapolate=False)
"""Splat an image with respect to a deformation field (pull adjoint).

Parameters
----------
input : (..., [channel], *inshape) tensor
    Input image.
grid : (..., *inshape, dim) tensor
    Transformation field.
shape : sequence[int], default=`inshape`
    Output shape
interpolation : int or sequence[int], default=1
    Interpolation order.
bound : str, or sequence[str], default='zero'
    Boundary conditions.
extrapolate : bool or int, default=True
    Extrapolate out-of-bound data.

Returns
-------
output : (..., [channel], *shape) tensor
    Spatted image.
"""
```

We also implement a convenience function that is equivalent to pushing a 
tensor of ones:

```python
grid_count(grid, shape=None, 
           interpolation='linear', bound='zero', extrapolate=False)
"""Splatting weights with respect to a deformation field (pull adjoint).

Parameters
----------
grid : (..., *inshape, dim) tensor
    Transformation field.
shape : sequence[int], default=`inshape`
    Output shape
interpolation : int or sequence[int], default=1
    Interpolation order.
bound : str, or sequence[str], default='zero'
    Boundary conditions.
extrapolate : bool or int, default=True
    Extrapolate out-of-bound data.

Returns
-------
output : (..., [1], *shape) tensor
    Splatted weights.
"""
```


## Solving large regularized linear systems

In imaging inverse problems, it is common to use regularizers that penalize the 
(voxelwise) l2 norm of some components of the spatial derivatives of a dense 
field. Consequently, inversion often consists of solving a linear system of 
equations of the form (**H** + **L**)<sup>-1</sup>(**g** + **Lv**), where 
**H** is block-diagonal, and **L** = **K**<sup>T</sup>**K** encodes the 
regularizer. This section implements regularizers and solvers for such problems.

The regularizers that we implement are identical to those implemented in 
[SPM](https://github.com/spm/spm12). They are a combination of three 
(for scalar fields) or five (for vector fields that encode dense displacements 
or velocities) energies. For dense scalar fields, we implement:
- the absolute energy ∫ ||f(**x**)||<sup>2</sup> d**x**
- the membrane energy ∫ ||∇ f(**x**)||<sup>2</sup> d**x**
- the bending energy ∫ ||∇<sup>2</sup> f(**x**)||<sub>F</sub><sup>2</sup> d**x**

For vector fields, we further implement two components of the linear-elastic 
energy, known as the Lamé coefficients:
- the trace of the strain tensor ∫ tr(**Df**(**x**)) d**x**
- the shear modulus ∫ ||**Df**(**x**) + **Df**(**x**)<sup>T</sup>||<sub>F</sub><sup>2</sup>  d**x** 

As opposed to SPM, we have the possibility to include a voxel-wise modulation of these energies, 
in which case the regularizer can be written as **L** = **K**<sup>T</sup>**WK**, where **W** is
diagonal. This voxel-wise modulation can be used to implement spatially varying regularization, 
or edge-preserving regularizers under an iteratively reweighted scheme.

Note that **none of the functions in this section are automatically 
differentiable**.

```python
regulariser(input, weight=None, hessian=None, dim=None,
            absolute=0, membrane=0, bending=0, factor=1,
            voxel_size=1, bound='dct2', output=None)
"""Apply the forward pass of a regularised linear system
        output = (hessian + regulariser) @ input

Parameters
----------
input : (N, C, *shape) tensor
weight : (N, C|1, *shape) tensor, optional
hessian : (N, CC, *shape) tensor, optional
    CC is one of {1, C, C*(C+1)/2}
dim : int, optional
absolute : [sequence of] float, default=0
membrane : [sequence of] float, default=0
bending : [sequence of] float, default=0
voxel_size : [sequence of] float, default=1.
bound : [sequence of] bound_like, default='dct2'
output : (N, C, *shape) tensor, optional

Returns
-------
output : (N, C, *shape) tensor
"""
```

```python
relax(gradient, weight=None, hessian=None, dim=None,
      absolute=0, membrane=0, bending=0, factor=1,
      voxel_size=1, bound='dct2', nb_iter=2, output=None)
"""Solve a regularised linear system by relaxation (Gauss-Seidel)
        solution = (hessian + regulariser) \ gradient

Parameters
----------
gradient : (N, C, *shape) tensor
weight : (N, C|1, *shape) tensor, optional
hessian : (N, CC, *shape) tensor, optional
    CC is one of {1, C, C*(C+1)/2}
absolute : [sequence of] float, default=0
membrane : [sequence of] float, default=0
bending : [sequence of] float, default=0
voxel_size : [sequence of] float, default=1.
bound : [sequence of] bound_like, default='dct2'
nb_iter : int, default=2
output : (N, C, *shape) tensor, optional

Returns
-------
output : (N, C, *shape) tensor
"""
```

```python
precond(gradient, weight=None, hessian=None, dim=None,
        absolute=0, membrane=0, bending=0, factor=1,
        voxel_size=1, bound='dct2', output=None)
"""Apply the preconditioner of  a regularised linear system
        solution = (inv(hessian) + diag(regulariser)) \ gradient

Parameters
----------
gradient : (N, C, *shape) tensor
weight : (N, C|1, *shape) tensor, optional
hessian : (N, CC, *shape) tensor, optional
    CC is one of {1, C, C*(C+1)/2}
absolute : [sequence of] float, default=0
membrane : [sequence of] float, default=0
bending : [sequence of] float, default=0
voxel_size : [sequence of] float, default=1.
bound : [sequence of] bound_like, default='dct2'
output : (N, C, *shape) tensor, optional

Returns
-------
output : (N, C, *shape) tensor
"""
```

```python
pcg(gradient, weight=None, hessian=None, dim=None,
    absolute=0, membrane=0, bending=0, factor=1,
    voxel_size=1, bound='dct2', nb_iter=16, tol=1e-4, output=None)
"""Solve a regularised linear system by conjugate gradient
        solution = (hessian + regulariser) \ gradient

Parameters
----------
gradient : (N, C, *shape) tensor
weight : (N, C|1, *shape) tensor, optional
hessian : (N, CC, *shape) tensor, optional
    CC is one of {1, C, C*(C+1)/2}
absolute : [sequence of] float, default=0
membrane : [sequence of] float, default=0
bending : [sequence of] float, default=0
voxel_size : [sequence of] float, default=1.
bound : [sequence of] bound_like, default='dct2'
nb_iter : int, default=2
output : (N, C, *shape) tensor, optional

Returns
-------
output : (N, C, *shape) tensor
"""
```

```python
fmg(hessian, gradient, weight=None, dim=None,
    absolute=0, membrane=0, bending=0, factor=1,
    voxel_size=1, bound='dct2',
    nb_cycles=2, nb_iter=2, max_levels=16,
    solver='cg', output=None)
"""Solve a regularised linear system by full multi-grid
        solution = (hessian + regulariser) \ gradient

Parameters
----------
hessian : (N, CC, *shape) tensor
    CC is one of {1, C, C*(C+1)/2}
gradient : (N, C, *shape) tensor
weight : (N, C|1, *shape) tensor, optional
absolute : [sequence of] float, default=0
membrane : [sequence of] float, default=0
bending : [sequence of] float, default=0
voxel_size : [sequence of] float, default=1.
bound : [sequence of] bound_like, default='dct2'
nb_cycles : int, default=2
nb_iter : int, default=4
max_levels : int, default=16
solver : {'relax', 'cg'}, default='relax'
output : (N, C, *shape) tensor, optional

Returns
-------
output : (N, C, *shape) tensor
"""
```


Versions of these functions specialized for vector fields are also implemented:

```python
regulariser_grid(input, weight=None, hessian=None,
                 absolute=0, membrane=0, bending=0, lame=0, factor=1,
                 voxel_size=1, bound='dft', output=None)
"""Apply the forward pass of a regularised linear system
        output = (hessian + regulariser) @ input

Parameters
----------
input : (N, *shape, D) tensor
weight : (N, *shape) tensor, optional
hessian : (N, *shape, DD) tensor, optional
    DD is one of {1, D, D*(D+1)/2}
absolute : float, default=0
membrane : float, default=0
bending : float, default=0
lame : (float, float), default=0
voxel_size : [sequence of] float, default=1.
bound : [sequence of] bound_like, default='dft'
output : (N, *shape, D) tensor, optional

Returns
-------
output : (N, *shape, D) tensor

"""
```

```python
relax_grid(gradient, weight=None, hessian=None,
           absolute=0, membrane=0, bending=0, lame=0, factor=1,
           voxel_size=1, bound='dft', nb_iter=2, output=None)
"""Solve a regularised linear system by relaxation (Gauss-Seidel)
        solution = (hessian + regulariser) \ gradient

Parameters
----------
gradient : (N, *shape, D) tensor
weight : (N, *shape) tensor, optional
hessian : (N, *shape, DD) tensor, optional
    DD is one of {1, D, D*(D+1)/2}
absolute : float, default=0
membrane :  float, default=0
bending :  float, default=0
lame : (float, float), default=0
voxel_size : [sequence of] float, default=1.
bound : [sequence of] bound_like, default='dft'
nb_iter : int, default=2
output : (N, *shape, D) tensor, optional

Returns
-------
output : (N, *shape, D) tensor
"""
```

```python
precond_grid(gradient, weight=None, hessian=None,
             absolute=0, membrane=0, bending=0, lame=0, factor=1,
             voxel_size=1, bound='dft', output=None)
"""Apply the preconditioner of  a regularised linear system
        solution = (inv(hessian) + diag(regulariser)) \ gradient

Parameters
----------
gradient : (N, *shape, D) tensor
weight : (N, *shape) tensor, optional
hessian : (N, *shape, DD) tensor, optional
    DD is one of {1, D, D*(D+1)/2}
absolute : float, default=0
membrane :  float, default=0
bending :  float, default=0
lame : (float, float), default=0
voxel_size : [sequence of] float, default=1.
bound : [sequence of] bound_like, default='dft'
output : (N, *shape, D) tensor, optional

Returns
-------
output : (N, *shape, D) tensor

"""
```

```python
pcg_grid(gradient, weight=None, hessian=None,
         absolute=0, membrane=0, bending=0, lame=0, factor=1,
         voxel_size=1, bound='dft', nb_iter=16, tol=1e-4, output=None)
"""Solve a regularised linear system by conjugate gradient
        solution = (hessian + regulariser) \ gradient

Parameters
----------
gradient : (N, *shape, D) tensor
weight : (N, C*shape) tensor, optional
hessian : (N, *shape, DD) tensor, optional
    DD is one of {1, D, D*(D+1)/2}
absolute : [sequence of] float, default=0
membrane : [sequence of] float, default=0
bending : [sequence of] float, default=0
lame : (float, float), default=0
voxel_size : [sequence of] float, default=1.
bound : [sequence of] bound_like, default='dft'
nb_iter : int, default=2
output : (N, *shape, D) tensor, optional

Returns
-------
output : (N, *shape, D) tensor

"""
```

```python
fmg_grid(hessian, gradient, weight=None,
         absolute=0, membrane=0, bending=0, lame=0, factor=1,
         voxel_size=1, bound='dft',
         nb_cycles=2, nb_iter=2, max_levels=16,
         solver='cg', output=None)
"""Solve a regularised linear system by full multi-grid
        solution = (hessian + regulariser) \ gradient

Parameters
----------
hessian : (N, *shape, DD) tensor
    DD is one of {1, D, D*(D+1)/2}
gradient : (N, *shape, D) tensor
weight : (N, *shape) tensor, optional
absolute : float, default=0
membrane : float, default=0
bending : float, default=0
lame : (float, float), default=0
voxel_size : [sequence of] float, default=1.
bound : [sequence of] bound_like, default='dft'
nb_cycles : int, default=2
nb_iter : int, default=4
max_levels : int, default=16
solver : {'relax', 'cg'}, default='relax'
output : (N, *shape, D) tensor, optional

Returns
-------
output : (N, *shape, D) tensor

"""
```


## Valid combinations of Python / PyTorch / CUDA

We only list combinations for which fastfields is also compiled 
(e.g., pytorch 1.4 is available on Python 2.7, but we do not list it because 
fastfields is not available on Python 2).

Note that fastfields requires PyTorch >= 1.4.


| **PyTorch:**| 1.4.0 | 1.5.0 | 1.5.1 | 1.6.0 | 1.7.0 | 1.7.1 | 1.8.0 | 1.8.1 | 1.9.0 | 1.9.1 | 1.10.0 | 1.10.1 | 1.10.2 | 1.11.0 | 1.12.0 |
| **Python:** | 3.5-8 | 3.5-8 | 3.6-8 | 3.6-8 | 3.6-8 | 3.6-9 | 3.6-9 | 3.6-9 | 3.6-9 | 3.6-9 | 3.6-9  | 3.6-9  | 3.6-9  | 3.7-10 | 3.7-10 |
|-------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:------:|:------:|:------:|:------:|:------:|
| cpu         |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓    |   ✓    |   ✓    |   ✓    |   ✓    |
| 9.2         |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |       |       |       |       |        |        |        |        |        |
| 10.0        |   ✓   |       |       |       |       |       |       |       |       |       |        |        |        |        |        |
| 10.1        |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |       |       |        |        |        |        |        |
| 10.2        |       |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓    |   ✓    |   ✓    |   ✓    |   ✓    |
| 11.0        |       |       |       |       |   ✓   |   ✓   |       |       |       |       |        |        |        |        |        |
| 11.1        |       |       |       |       |       |       |   ✓   |   ✓   |   ✓   |   ✓   |   ✓    |   ✓    |   ✓    |        |        |
| 11.3        |       |       |       |       |       |       |       |       |       |       |   ✓    |   ✓    |   ✓    |   ✓    |   ✓    |
| 11.5        |       |       |       |       |       |       |       |       |       |       |        |        |        |   ✓    |        |
| 11.6        |       |       |       |       |       |       |       |       |       |       |        |        |        |        |   ✓    |

