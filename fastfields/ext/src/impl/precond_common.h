#pragma once

#include <ATen/ATen.h>
#include "../bounds.h"

#define FF_PRECOND_DECLARE(space) \
  namespace space { \
    at::Tensor precond_impl( \
      at::Tensor hessian, const at::Tensor& gradient, at::Tensor solution, at::Tensor weight, \
      c10::ArrayRef<double> absolute, c10::ArrayRef<double> membrane, c10::ArrayRef<double> bending, \
      c10::ArrayRef<double> voxel_size, BoundVectorRef bound); \
  }


namespace ff {
FF_PRECOND_DECLARE(cpu)
FF_PRECOND_DECLARE(cuda)
FF_PRECOND_DECLARE(notimplemented)
} // namespace ff
