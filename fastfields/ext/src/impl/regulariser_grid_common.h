#pragma once

#include <ATen/ATen.h>
#include "../bounds.h"

#define FF_REGULARISER_GRID_DECLARE(space) \
  namespace space { \
    at::Tensor regulariser_grid_impl( \
      const at::Tensor& input, at::Tensor output, at::Tensor weight, at::Tensor hessian, \
      double absolute, double membrane, double bending, double lame_shear, double lame_div, \
      c10::ArrayRef<double> voxel_size, BoundVectorRef bound); \
  }


namespace ff {
FF_REGULARISER_GRID_DECLARE(cpu)
FF_REGULARISER_GRID_DECLARE(cuda)
FF_REGULARISER_GRID_DECLARE(notimplemented)
} // namespace ff
