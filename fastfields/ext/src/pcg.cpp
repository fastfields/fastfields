#include "impl/pcg_common.h"
#include "checks.h"
#include "bounds.h"
#include <ATen/ATen.h>
#include <torch/utils.h>  // NoGradGuard
#include <vector>
#include <utility>

#ifndef FF_WITH_CUDA
#  define cuda notimplemented
#endif

using at::Tensor;
using c10::IntArrayRef;
using c10::ArrayRef;
using std::vector;



namespace ff {

Tensor pcg(const Tensor & hessian, 
           const Tensor & gradient,
           const Tensor & solution,
           const Tensor & weight,
           const vector<double> &  absolute, 
           const vector<double> &  membrane, 
           const vector<double> &  bending,
           const vector<double> &  voxel_size, 
           const vector<BoundType> & bound,
           int64_t nb_iter, double tol) 
{
  torch::NoGradGuard no_grad;

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
    return cuda::pcg_impl(hessian, gradient, solution, weight,
        ArrayRef<double>(absolute), ArrayRef<double>(membrane), ArrayRef<double>(bending),
        ArrayRef<double>(voxel_size), BoundVectorRef(bound), nb_iter, tol);
  else
    return cpu::pcg_impl(hessian, gradient, solution, weight,
        ArrayRef<double>(absolute), ArrayRef<double>(membrane), ArrayRef<double>(bending),
        ArrayRef<double>(voxel_size), BoundVectorRef(bound), nb_iter, tol);
}

Tensor pcg_grid(const Tensor & hessian, 
                const Tensor & gradient,
                const Tensor & solution,
                const Tensor & weight,
                      double    absolute, 
                      double    membrane, 
                      double    bending,
                      double    lame_shear,
                      double    lame_div,
                const vector<double> & voxel_size, 
                const vector<BoundType> & bound,
                int64_t nb_iter, double tol)
{
  torch::NoGradGuard no_grad;

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
    return cuda::pcg_grid_impl(hessian, gradient, solution, weight,
        absolute, membrane, bending, lame_shear, lame_div,
        ArrayRef<double>(voxel_size), BoundVectorRef(bound), nb_iter, tol);
  else
    return cpu::pcg_grid_impl(hessian, gradient, solution, weight,
        absolute, membrane, bending, lame_shear, lame_div,
        ArrayRef<double>(voxel_size), BoundVectorRef(bound), nb_iter, tol);
}

} // namespace ff
