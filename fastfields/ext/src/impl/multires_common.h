#pragma once

#include <ATen/ATen.h>
#include "../bounds.h"


#define FF_MULTIRES_DECLARE(space) \
  namespace space { \
    at::Tensor multires_impl( \
      at::Tensor source, at::Tensor target, \
      c10::ArrayRef<double> factor, BoundVectorRef bound,  \
      bool do_adjoint); \
  }


namespace ff {
FF_MULTIRES_DECLARE(cpu)
FF_MULTIRES_DECLARE(cuda)
FF_MULTIRES_DECLARE(notimplemented)
} // namespace ff
