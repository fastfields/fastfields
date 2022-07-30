#pragma once

#include <ATen/ATen.h>
#include "../bounds.h"
#include "../interpolation.h"
#include "../grid_align.h"


#define FF_RESIZE_DECLARE(space) \
  namespace space { \
    at::Tensor resize_impl( \
      at::Tensor source, at::Tensor target, \
      c10::ArrayRef<double> factor, BoundVectorRef bound,  \
      InterpolationVectorRef interpolation, GridAlignVectorRef mode, \
      bool do_adjoint, bool normalize); \
  }


namespace ff {
FF_RESIZE_DECLARE(cpu)
FF_RESIZE_DECLARE(cuda)
FF_RESIZE_DECLARE(notimplemented)
} // namespace ff
