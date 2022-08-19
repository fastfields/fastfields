#pragma once

#include <ATen/ATen.h>
#include "../bounds.h"
#include "../interpolation.h"
#include <deque>

#define FF_PUSHPULL_DECLARE(space) \
  namespace space { \
    template <typename SourceType> \
    std::deque<at::Tensor> pushpull( \
      const SourceType& source, const at::Tensor& grid, \
      BoundVectorRef bound, InterpolationVectorRef interpolation, int extrapolate, \
      bool do_pull, bool do_push, bool do_count, bool do_grad, bool do_sgrad); \
    template <typename SourceType> \
    std::deque<at::Tensor> pushpull( \
      const SourceType & source, const at::Tensor& grid, const at::Tensor& target, \
      BoundVectorRef bound, InterpolationVectorRef interpolation, int extrapolate, \
      bool do_pull, bool do_push, bool do_count, bool do_grad, bool do_sgrad); \
  }

namespace ff {
FF_PUSHPULL_DECLARE(cpu)
FF_PUSHPULL_DECLARE(cuda) 
FF_PUSHPULL_DECLARE(notimplemented)
} // namespace ff
