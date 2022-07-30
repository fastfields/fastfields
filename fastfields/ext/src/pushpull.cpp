#include "impl/pushpull_common.h"
#include "checks.h"
#include <ATen/ATen.h>
#include <vector>
#include <deque>
#include <iostream>


#ifndef FF_WITH_CUDA
#  define cuda notimplemented
#endif


using at::Tensor;
using c10::IntArrayRef;

namespace ff {


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PULL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor grid_pull(const Tensor& input, const Tensor& grid,
                 const std::vector<BoundType> & bound_mode, 
                 const std::vector<InterpolationType> & interpolation_mode, 
                 int extrapolate)  {

  FF_CHECK_DEFINED(input)
  FF_CHECK_DEFINED(grid)
  auto input_opt = input.options();
  auto grid_opt  = grid.options();
  FF_CHECK_OPT_STRIDED(input_opt)
  FF_CHECK_OPT_STRIDED(grid_opt)
  FF_CHECK_OPT_SAME_DEVICE(input_opt, grid_opt)
  FF_CHECK_OPT_SAME_DTYPE(input_opt, grid_opt)
  FF_CHECK_1D_2D_OR_3D(input)
  FF_CHECK_1D_2D_OR_3D(grid)
  FF_CHECK_GRID_COMPONENT(grid, grid.dim())
  FF_CHECK_NOT_EMPTY(input)
  FF_CHECK_NOT_EMPTY(grid)
  FF_CHECK_VEC_NOT_EMPTY(bound_mode);
  FF_CHECK_VEC_NOT_EMPTY(interpolation_mode);

  if (input.is_cuda())
    return cuda::pushpull(input, grid,
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/true, /*push*/false, /*count*/false,
      /*grad*/false, /*sgrad*/false).front();
  else
    return cpu::pushpull(input, grid, 
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/true, /*push*/false, /*count*/false,
      /*grad*/false, /*sgrad*/false).front();
}

std::deque<Tensor>
grid_pull_backward(const Tensor& grad, const Tensor& input, const Tensor& grid,
                   const std::vector<BoundType> & bound_mode, 
                   const std::vector<InterpolationType> & interpolation_mode, 
                   int extrapolate)
{
  if (input.is_cuda()) {
    return cuda::pushpull(input, grid, grad,
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/false,
      /*push*/input.requires_grad(), /*count*/false,
      /*grad*/grid.requires_grad(), /*sgrad*/false);
  } else {
    return cpu::pushpull(input, grid, grad,
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/false,
      /*push*/input.requires_grad(), /*count*/false,
      /*grad*/grid.requires_grad(), /*sgrad*/false);
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PUSH ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor grid_push(const Tensor& input, const Tensor& grid,
                 IntArrayRef source_size,
                 const std::vector<BoundType> & bound_mode, 
                 const std::vector<InterpolationType> & interpolation_mode, 
                 int extrapolate) {

  FF_CHECK_DEFINED(input)
  FF_CHECK_DEFINED(grid)
  auto input_opt = input.options();
  auto grid_opt  = grid.options();
  FF_CHECK_OPT_STRIDED(input_opt)
  FF_CHECK_OPT_STRIDED(grid_opt)
  FF_CHECK_OPT_SAME_DEVICE(input_opt, grid_opt)
  FF_CHECK_OPT_SAME_DTYPE(input_opt, grid_opt)
  FF_CHECK_1D_2D_OR_3D(input)
  FF_CHECK_1D_2D_OR_3D(grid)
  FF_CHECK_GRID_COMPONENT(grid, grid.dim())
  FF_CHECK_NOT_EMPTY(input)
  FF_CHECK_NOT_EMPTY(grid)
  FF_CHECK_GRID_TARGET_COMPAT(grid, input)
  FF_CHECK_VEC_NOT_EMPTY(bound_mode);
  FF_CHECK_VEC_NOT_EMPTY(interpolation_mode);

  if (source_size.empty())
  {
    auto size = IntArrayRef({input.dim() >= 3 ? input.size(2) : 1,
                             input.dim() >= 4 ? input.size(3) : 1,
                             input.dim() >= 5 ? input.size(4) : 1});
    if (input.is_cuda())
      return cuda::pushpull(size, grid, input,
        BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
        extrapolate, /*pull*/false, /*push*/true, /*count*/false,
        /*grad*/false, /*sgrad*/false).front();
    else
      return cpu::pushpull(size, grid, input,
        BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
        extrapolate, /*pull*/false, /*push*/true, /*count*/false,
        /*grad*/false, /*sgrad*/false).front();
  } 
  else 
  {
    FF_CHECK_VEC_LENGTH(source_size, grid.dim())
    if (input.is_cuda())
      return cuda::pushpull(source_size, grid, input,
        BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
        extrapolate, /*pull*/false, /*push*/true, /*count*/false,
        /*grad*/false, /*sgrad*/false).front();
    else
      return cpu::pushpull(source_size, grid, input,
        BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
        extrapolate, /*pull*/false, /*push*/true, /*count*/false,
        /*grad*/false, /*sgrad*/false).front();

  }
}

std::deque<Tensor>
grid_push_backward(const Tensor& grad, const Tensor& input, const Tensor& grid,
                   const std::vector<BoundType> & bound_mode, 
                   const std::vector<InterpolationType> & interpolation_mode, 
                   int extrapolate)
{
  if (input.is_cuda()) {
    return cuda::pushpull(grad, grid, input,
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/input.requires_grad(), /*push*/false, /*count*/false,
      /*grad*/grid.requires_grad(), /*sgrad*/false);
  } else {
    return cpu::pushpull(grad, grid, input,
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/input.requires_grad(), /*push*/false, /*count*/false,
      /*grad*/grid.requires_grad(), /*sgrad*/false);
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COUNT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor grid_count(const Tensor& grid,
                 IntArrayRef source_size,
                 const std::vector<BoundType> & bound_mode, 
                 const std::vector<InterpolationType> & interpolation_mode, 
                 int extrapolate) {

  FF_CHECK_DEFINED(grid)
  auto grid_opt  = grid.options();
  FF_CHECK_OPT_STRIDED(grid_opt)
  FF_CHECK_1D_2D_OR_3D(grid)
  FF_CHECK_GRID_COMPONENT(grid, grid.dim())
  FF_CHECK_NOT_EMPTY(grid)
  FF_CHECK_VEC_NOT_EMPTY(bound_mode);
  FF_CHECK_VEC_NOT_EMPTY(interpolation_mode);

  if (source_size.empty())
  {
    auto size = IntArrayRef({grid.dim() >= 3 ? grid.size(2) : 1,
                             grid.dim() >= 4 ? grid.size(3) : 1,
                             grid.dim() >= 5 ? grid.size(4) : 1});
    if (grid.is_cuda())
      return cuda::pushpull(size, grid,
        BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
        extrapolate, /*pull*/ false, /*push*/ false, /*count*/ true,
        /*grad*/ false, /*sgrad*/ false).front();
    else
      return cpu::pushpull(size, grid,
        BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
        extrapolate, /*pull*/ false, /*push*/ false, /*count*/ true,
        /*grad*/ false, /*sgrad*/ false).front();
  } 
  else 
  {
    FF_CHECK_VEC_LENGTH(source_size, grid.dim())
    if (grid.is_cuda())
      return cuda::pushpull(source_size, grid,
        BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
        extrapolate, /*pull*/ false, /*push*/ false, /*count*/ true,
        /*grad*/ false, /*sgrad*/ false).front();
    else
      return cpu::pushpull(source_size, grid,
        BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
        extrapolate, /*pull*/ false, /*push*/ false, /*count*/ true,
        /*grad*/ false, /*sgrad*/ false).front();

  }
}

Tensor
grid_count_backward(const Tensor& grad, const Tensor& grid,
                    const std::vector<BoundType> & bound_mode, 
                    const std::vector<InterpolationType> & interpolation_mode, 
                    int extrapolate)
{
  if (grid.is_cuda()) {
    return cuda::pushpull(grad, grid,
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/ false, /*push*/ false, /*count*/ false,
      /*grad*/ grid.requires_grad(), /*sgrad*/ false).front();
  } else {
    return cpu::pushpull(grad, grid,
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/ false, /*push*/ false, /*count*/ false,
      /*grad*/ grid.requires_grad(), /*sgrad*/ false).front();
  }
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~ PULL GRADIENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor grid_grad(const Tensor& input, const Tensor& grid,
                 const std::vector<BoundType> & bound_mode, 
                 const std::vector<InterpolationType> & interpolation_mode, 
                 int extrapolate)  {

  FF_CHECK_DEFINED(input)
  FF_CHECK_DEFINED(grid)
  auto input_opt = input.options();
  auto grid_opt  = grid.options();
  FF_CHECK_OPT_STRIDED(input_opt)
  FF_CHECK_OPT_STRIDED(grid_opt)
  FF_CHECK_OPT_SAME_DEVICE(input_opt, grid_opt)
  FF_CHECK_OPT_SAME_DTYPE(input_opt, grid_opt)
  FF_CHECK_1D_2D_OR_3D(input)
  FF_CHECK_1D_2D_OR_3D(grid)
  FF_CHECK_GRID_COMPONENT(grid, grid.dim())
  FF_CHECK_NOT_EMPTY(input)
  FF_CHECK_NOT_EMPTY(grid)
  FF_CHECK_VEC_NOT_EMPTY(bound_mode);
  FF_CHECK_VEC_NOT_EMPTY(interpolation_mode);

  if (input.is_cuda())
    return cuda::pushpull(input, grid,
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode),
      extrapolate, /*pull*/false, /*push*/false, /*count*/false,
      /*grad*/false, /*sgrad*/true).front();
  else
    return cpu::pushpull(input, grid, 
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/false, /*push*/false, /*count*/false,
      /*grad*/false, /*sgrad*/true).front();
}

std::deque<Tensor>
grid_grad_backward(const Tensor& grad, const Tensor& input, const Tensor& grid,
                   const std::vector<BoundType> & bound_mode, 
                   const std::vector<InterpolationType> & interpolation_mode, 
                   int extrapolate)
{
  if (input.is_cuda()) {
    return cuda::pushpull(input, grid, grad,
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/false,  /*push*/input.requires_grad(),
      /*count*/false, /*grad*/grid.requires_grad(), /*sgrad*/false);
  } else {
    return cpu::pushpull(input, grid, grad,
      BoundVectorRef(bound_mode), InterpolationVectorRef(interpolation_mode), 
      extrapolate, /*pull*/false,  /*push*/input.requires_grad(),
      /*count*/false, /*grad*/grid.requires_grad(), /*sgrad*/false);
  }
}

} // namespace ff
