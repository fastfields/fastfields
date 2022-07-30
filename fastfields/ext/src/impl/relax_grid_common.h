#pragma once

#include <ATen/ATen.h>
#include "../bounds.h"

#define FF_RELAX_GRID_DECLARE(space) \
  namespace space { \
    at::Tensor relax_grid_impl( \
      at::Tensor hessian, const at::Tensor& gradient, at::Tensor solution, at::Tensor weight,  \
      double absolute, double membrane, double bending, double lame_shear, double lame_div, \
      at::ArrayRef<double> voxel_size, BoundVectorRef bound, int64_t nb_iter); \
  }


namespace ff {
FF_RELAX_GRID_DECLARE(cpu)
FF_RELAX_GRID_DECLARE(cuda)
FF_RELAX_GRID_DECLARE(notimplemented)
} // namespace ff
