#pragma once

#include <ATen/ATen.h>
#include "../bounds.h"

#define FF_REGULARISER_DECLARE(space) \
  namespace space { \
    at::Tensor regulariser_impl( \
      const at::Tensor& input, at::Tensor output, at::Tensor weight, at::Tensor hessian, \
      c10::ArrayRef<double> absolute, c10::ArrayRef<double> membrane, c10::ArrayRef<double> bending, \
      c10::ArrayRef<double> voxel_size, BoundVectorRef bound); \
  }

namespace ff {
FF_REGULARISER_DECLARE(cpu)
FF_REGULARISER_DECLARE(cuda)
FF_REGULARISER_DECLARE(notimplemented)
} // namespace ff
