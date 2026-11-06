# API reference

The `fastfields` distribution ships five subpackages of the `fastfields`
namespace. Pick the one that matches the arrays you work with, or use
`fastfields.auto` and let it choose.

| Page | Module | For |
|---|---|---|
| [auto](auto.md) | `fastfields.auto` | dispatches on the type of the array you pass |
| [numpy](numpy.md) | `fastfields.numpy` | NumPy arrays, on the CPU |
| [torch](torch.md) | `fastfields.torch` | PyTorch tensors, with autograd |
| [cupy](cupy.md) | `fastfields.cupy` | CuPy arrays in CUDA device memory |
| [helpers](helpers.md) | `fastfields.helpers` | the `Spline` and `Bound` enums and the argument normalisers |

The raw bindings that all of them are built on, `fastfields.dlpack`, ship from
a separate distribution (`fastfields-dlpack`) and are documented
[there](https://fastfields.github.io/dlpack/).
