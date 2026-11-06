# API reference

The `fastfields` distribution ships five subpackages of the `fastfields`
namespace. Pick the one matching the arrays you work with, or
`fastfields.auto` to let it dispatch for you.

| Page | Module | For |
|---|---|---|
| [auto](auto.md) | `fastfields.auto` | dispatches on the array type you pass |
| [numpy](numpy.md) | `fastfields.numpy` | NumPy arrays, CPU |
| [torch](torch.md) | `fastfields.torch` | PyTorch tensors, with autograd |
| [cupy](cupy.md) | `fastfields.cupy` | CuPy arrays in CUDA device memory |
| [helpers](helpers.md) | `fastfields.helpers` | the shared `Spline`/`Bound` enums and argument normalisers |

The raw bindings they are all built on, `fastfields.dlpack`, ship from a
separate distribution (`fastfields-dlpack`) and are documented
[there](https://fastfields.github.io/dlpack/).
