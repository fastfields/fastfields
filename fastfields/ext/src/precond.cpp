#include "impl/precond_common.h"
#include "checks.h"
#include <ATen/ATen.h>
#include <vector>
#include <utility>

#ifndef FF_WITH_CUDA
#  define cuda notimplemented
#endif

using at::Tensor;
using c10::IntArrayRef;
using c10::ArrayRef;



namespace ff {

Tensor precond(
    const Tensor& hessian, const Tensor& gradient,
    const Tensor& solution, const Tensor& weight,
    const std::vector<double> & absolute, const std::vector<double> & membrane, 
    const std::vector<double> & bending, const std::vector<double> & voxel_size, 
    const std::vector<BoundType> & bound) {

  FF_CHECK_DEFINED(gradient)
  auto gradient_opt = gradient.options();
  FF_CHECK_OPT_STRIDED(gradient_opt)
  FF_CHECK_1D_2D_OR_3D(gradient)
  FF_CHECK_NOT_EMPTY(gradient)
  FF_CHECK_VEC_NOT_EMPTY(bound);

  if (hessian.defined() && hessian.numel() > 0)
  {
    auto hessian_opt  = hessian.options();
    FF_CHECK_OPT_STRIDED(hessian_opt)
    FF_CHECK_OPT_SAME_DEVICE(gradient_opt, hessian_opt)
    FF_CHECK_OPT_SAME_DTYPE(gradient_opt, hessian_opt)
    FF_CHECK_1D_2D_OR_3D(hessian)
    FF_CHECK_NOT_EMPTY(hessian)
  }

  if (solution.defined() && solution.numel() > 0)
  {
    auto solution_opt  = solution.options();
    FF_CHECK_OPT_STRIDED(solution_opt)
    FF_CHECK_OPT_SAME_DEVICE(gradient_opt, solution_opt)
    FF_CHECK_OPT_SAME_DTYPE(gradient_opt, solution_opt)
    FF_CHECK_1D_2D_OR_3D(solution)
    FF_CHECK_NOT_EMPTY(solution)
  }

  if (weight.defined() && weight.numel() > 0)
  {
    auto weight_opt  = weight.options();
    FF_CHECK_OPT_STRIDED(weight_opt)
    FF_CHECK_OPT_SAME_DEVICE(gradient_opt, weight_opt)
    FF_CHECK_OPT_SAME_DTYPE(gradient_opt, weight_opt)
    FF_CHECK_1D_2D_OR_3D(weight)
    FF_CHECK_NOT_EMPTY(weight)
  }

  if (gradient.is_cuda())
    return cuda::precond_impl(hessian, gradient, solution, weight,
        ArrayRef<double>(absolute), ArrayRef<double>(membrane), ArrayRef<double>(bending),
        ArrayRef<double>(voxel_size), BoundVectorRef(bound));
  else
    return cpu::precond_impl(hessian, gradient, solution, weight,
        ArrayRef<double>(absolute), ArrayRef<double>(membrane), ArrayRef<double>(bending),
        ArrayRef<double>(voxel_size), BoundVectorRef(bound));
}

} // namespace ff
