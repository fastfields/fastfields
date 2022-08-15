#pragma once
#include "common.h"
#include "../defines.h"
#include "utils.h"
#include <cmath>

#define OnePlusTiny 1.000001;

namespace ff {
FF_NAMESPACE_DEVICE {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                                Enum
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

enum class HessianType: uint8_t {
  None,         // No Hessian provided so nothing to do
  Eye,          // Scaled identity
  Diag,         // Diagonal matrix
  ESTATICS,     // (C-1) elements are independent conditioned on the last one
  Sym           // Symmetric matrix
};

template <typename offset_t>
static FF_HOST FF_INLINE
HessianType guess_hessian_type(offset_t C, offset_t CC)
{
  if (CC == 0)
    return HessianType::None;
  else if (CC == 1)
    return HessianType::Eye;
  else if (CC == C)
    return HessianType::Diag;
  else if (CC == 2*C-1)
    return HessianType::ESTATICS;
  else
    return HessianType::Sym;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             Cholesky
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace Cholesky {

  // Cholesky decomposition (Cholesky–Banachiewicz)
  //
  // @param[in]     C:  (u)int
  // @param[inout]  a:  CxC matrix
  // @param         s:  stride
  //
  // https://en.wikipedia.org/wiki/Cholesky_decomposition
  template <typename reduce_t, typename offset_t> FF_INLINE FF_DEVICE static
  void decompose(offset_t C, reduce_t a[], offset_t s = static_cast<offset_t>(1))
  {
    reduce_t sm, sm0;

    sm0  = 1e-40;
  #if 0
    for(offset_t c = 0; c < C; ++c) sm0 += a[(c*C+c)*s];
    sm0 *= 1e-7;
    sm0 *= sm0;
  #endif

    for (offset_t c = 0; c < C; ++c)
    {
      for (offset_t b = c; b < C; ++b)
      {
        sm = a[(c*C+b)*s];
        for(offset_t d = c-1; d >= 0; --d)
          sm -= a[(c*C+d)*s] * a[(b*C+d)*s];
        if (c == b) {
          a[(c*C+c)*s] = std::sqrt(MAX(sm, sm0));
        } else
          a[(b*C+c)*s] = sm / a[(c*C+c)*s];
      }
    }
    return;
  }

  // Cholesky solver (inplace)
  // @param[in]    C:  (u)int
  // @param[in]    a:  CxC matrix
  // @param        sa: matrix stride
  // @param[inout] x:  C vector
  // @param        sx: vector stride
  template <typename reduce_t, typename offset_t> FF_INLINE FF_DEVICE static
  void solve(offset_t C, const reduce_t * a, offset_t sa, reduce_t * x, offset_t sx)
  {
    reduce_t sm;
    for (offset_t c = 0; c < C; ++c)
    {
      sm = x[c*sx];
      for (offset_t cc = c-1; cc >= 0; --cc)
        sm -= a[(c*C+cc)*sa] * x[cc*sx];
      x[c*sx] = sm / a[c*(C+1)*sa];
    }
    for(offset_t c = C-1; c >= 0; --c)
    {
      sm = x[c*sx];
      for(offset_t cc = c+1; cc < C; ++cc)
        sm -= a[(cc*C+c)*sa] * x[cc*sx];
      x[c*sx] = sm / a[c*(C+1)*sa];
    }
  }
  
} // namespace Cholesky

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                            Static traits
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Function common to all Hessian types
template <typename Child>
struct HessianCommon 
{
  /// Copy values into the flattened hessian
  /// @param o      pointer to output array
  /// @param so     output array stride
  /// @param i      pointer to input array
  /// @param si     input array stride
  template <typename scalar_t, typename offset_t, typename reduce_t>
  static FF_INLINE FF_DEVICE
  void set(offset_t C, scalar_t * o, offset_t so,
           const reduce_t * i, offset_t si = static_cast<offset_t>(1))
  {
    for (offset_t c = 0; c < C; ++c, o += so)
      *o = i[c*si];
  }

  /// Add values into the flattened hessian
  /// @param o      pointer to output array
  /// @param so     output array stride
  /// @param i      pointer to input array
  /// @param si     input array stride
  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static FF_INLINE FF_DEVICE 
  void add(offset_t C, scalar_t * o, offset_t so,
           const reduce_t * i, offset_t si = static_cast<offset_t>(1))
  {
    for (offset_t c = 0; c < C; ++c, o += so)
      *o += i[c*si];
  }

  /// Solve the linear system: x = (H + diag(w))\v
  /// @param C      number of channels
  /// @param x      pointer to output array
  /// @param sx     output array stride
  /// @param h      pointer to hessian
  /// @param sh     hessian stride
  /// @param v      pointer to value at which to solve
  /// @param sv     value stride
  /// @param m      pointer to temporary buffer that stores the full matrix
  /// @param sm     buffer stride
  /// @param w      pointer to regulariser weights
  /// @param sw     weights stride
  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static FF_DEVICE 
  void invert(offset_t C, 
              scalar_t * x, offset_t sx, const scalar_t * h, offset_t sh,
              reduce_t * v, offset_t sv, reduce_t * m, offset_t sm,
              const reduce_t * w = static_cast<const reduce_t*>(nullptr),
              offset_t sw = static_cast<offset_t>(1))
  {
    Child::get(C, h, sh, m, sm);
    Child::invert_(C, m, sm, v, sv, w, sw);
    set(C, x, sx, v, sv);
  }

  /// Solve the linear system and increment x += (H + diag(w))\v
  /// @param C      number of channels
  /// @param x      pointer to output array
  /// @param sx     output array stride
  /// @param h      pointer to hessian
  /// @param sh     hessian stride
  /// @param v      pointer to value at which to solve
  /// @param sv     value stride
  /// @param m      pointer to temporary buffer that stores the full matrix
  /// @param sm     buffer stride
  /// @param w      pointer to regulariser weights
  /// @param sw     weights stride
  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static FF_DEVICE 
  void addinvert(offset_t C, 
                 scalar_t * x, offset_t sx, const scalar_t * h, offset_t sh,
                 reduce_t * v, offset_t sv, reduce_t * m, offset_t sm,
                 const reduce_t * w = static_cast<const reduce_t*>(nullptr),
                 offset_t sw = static_cast<offset_t>(1))
  {
    Child::get(C, h, sh, m, sm);
    Child::submatvec_(C, x, sx, m, sm, v, sv);
    Child::invert_(C, m, sm, v, sv, w, sw);
    add(C, x, sx, v, sv);
  }
};

template <HessianType hessian_t>
struct HessianUtils: HessianCommon<HessianUtils<hessian_t> >
{};

// aliases to make the following code less ugly
using HessianUtilsNone  = HessianUtils<HessianType::None>;
using HessianUtilsEye   = HessianUtils<HessianType::Eye>;
using HessianUtilsDiag  = HessianUtils<HessianType::Diag>;
using HessianUtilsEst   = HessianUtils<HessianType::ESTATICS>;
using HessianUtilsSym   = HessianUtils<HessianType::Sym>;
using HessianCommonNone = HessianCommon<HessianUtilsNone>;
using HessianCommonEye  = HessianCommon<HessianUtilsEye>;
using HessianCommonDiag = HessianCommon<HessianUtilsDiag>;
using HessianCommonEst  = HessianCommon<HessianUtilsEst>;
using HessianCommonSym  = HessianCommon<HessianUtilsSym>;

template <>
struct HessianUtils<HessianType::None>: HessianCommonNone
{
  template <typename offset_t>
  static FF_INLINE offset_t work_size(offset_t C) { return 0; }

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static FF_INLINE FF_DEVICE void 
  get(offset_t C, const scalar_t * i, offset_t s, reduce_t * o, offset_t so)
  {}

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static FF_INLINE FF_DEVICE void 
  submatvec_(offset_t C, const scalar_t * i, offset_t s,
             const reduce_t * h, offset_t sh, reduce_t * o, offset_t so)
  {}

  template <typename offset_t, typename reduce_t>
  static FF_INLINE FF_DEVICE void 
  invert_(offset_t C,
          reduce_t * h, offset_t sh, reduce_t * v, offset_t sv,
          const reduce_t * w = static_cast<const reduce_t*>(nullptr),
          offset_t sw = static_cast<offset_t>(1)) {
    for (offset_t c = 0; c < C; ++c, v += sv, w += sw)
      (*v) /= (*w);
  }

  // specialize parent functions to avoid defining zero-sized arrays

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static FF_DEVICE 
  void invert(offset_t C, 
              scalar_t * x, offset_t sx, const scalar_t * h, offset_t sh,
              reduce_t * v, offset_t sv, reduce_t * m, offset_t sm,
              const reduce_t * w, offset_t sw)
  {
    get(C, h, sh, m, sm);
    invert_(C, m, sm, v, sv, w, sw);
    HessianCommonNone::set(C, x, sx, v, sv);
  }

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static FF_DEVICE 
  void addinvert(offset_t C, 
                 scalar_t * x, offset_t sx, const scalar_t * h, offset_t sh,
                 reduce_t * v, offset_t sv, reduce_t * m, offset_t sm,
                 const reduce_t * w = static_cast<const reduce_t*>(nullptr),
                 offset_t sw = static_cast<offset_t>(1))
  {
    m = nullptr;
    sm = 0;
    get(C, h, sh, m, sm);
    invert_(C, m, sm, v, sv, w, sw);
    HessianCommonNone::add(C, x, sx, v, sv);
  }
};

template <>
struct HessianUtils<HessianType::Eye>: HessianCommonEye
{
  template <typename offset_t>
  static FF_INLINE offset_t work_size(offset_t C) { return 1; }

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static FF_INLINE FF_DEVICE void 
  get(offset_t C, const scalar_t * i, offset_t si, reduce_t * o, offset_t so)
  {
    *o = static_cast<reduce_t>(*i);
  }

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static FF_INLINE FF_DEVICE void 
  submatvec_(offset_t C, const scalar_t * i, offset_t si,
             const reduce_t * h, offset_t sh, reduce_t * o, offset_t so)
  {
    reduce_t hh = *h;
    for (offset_t c = 0; c < C; ++c, i += si, o += so)
      (*o) -= hh * (*i);
  }

  template <typename reduce_t, typename offset_t>
  static FF_INLINE FF_DEVICE void 
  invert_(offset_t C, reduce_t * h, offset_t sh, reduce_t * v, offset_t sv,
          const reduce_t * w = static_cast<const reduce_t*>(nullptr),
          offset_t sw = static_cast<offset_t>(1)) {
    reduce_t hh = *h;
    for (offset_t c = 0; c < C; ++c, v += sv, w += sw)
      (*v) /= hh + (*w);
  }
};

template <>
struct HessianUtils<HessianType::Diag>: HessianCommonDiag
{
  template <typename offset_t>
  static FF_INLINE offset_t work_size(offset_t C) { return C; }

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static FF_INLINE FF_DEVICE void 
  get(offset_t C, const scalar_t * i, offset_t si, reduce_t * o, offset_t so)
  {
    for (offset_t c = 0; c < C; ++c, i += si, o += so)
      (*o) = static_cast<reduce_t>(*i);
  }

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static FF_INLINE FF_DEVICE void 
  submatvec_(offset_t C, const scalar_t * i, offset_t si,
             const reduce_t * h, offset_t sh, reduce_t * o, offset_t so)
  {
    for (offset_t c = 0; c < C; ++c, i += si, o += so, h += sh)
      (*o) -= (*h) * (*i);
  }

  template <typename reduce_t, typename offset_t>
  static FF_INLINE FF_DEVICE void 
  invert_(offset_t C, reduce_t * h, offset_t sh, reduce_t * v, offset_t sv,
          const reduce_t * w = static_cast<const reduce_t*>(nullptr),
          offset_t sw = static_cast<offset_t>(1)) {
    for (offset_t c = 0; c < C; ++c, v += sv, h += sh, w += sw)
      (*v) /= (*h) + (*w);
  }
};

template <>
struct HessianUtils<HessianType::ESTATICS>: HessianCommonEst
{
  template <typename offset_t>
  static FF_INLINE offset_t work_size(offset_t C) { return 2*C-1; }

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static FF_INLINE FF_DEVICE void 
  get(offset_t C, const scalar_t * i, offset_t si, reduce_t * o, offset_t so)
  {
    for (offset_t c = 0; c < 2*C-1; ++c, i += si, o += so)
      *o = static_cast<reduce_t>(*i);
  }

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static FF_INLINE FF_DEVICE void 
  submatvec_(offset_t C, const scalar_t * i, offset_t si,
             const reduce_t * h, offset_t sh, reduce_t * o, offset_t so)
  {
    const reduce_t * hh = h + C * sh;   // pointer to off-diagonal elements
    reduce_t * oo = o + (C-1) * so;     // pointer to last output element
    scalar_t r = i[(C-1)*si];
    for (offset_t c = 0; c < C-1; ++c, i += si, o += so, h += sh, hh += sh) {
      (*o) -= (*h) * (*i) + (*hh) * r;
      (*oo) -= (*hh) * (*i);
    }
    (*o) -= r * (*h);
  }

  template <typename reduce_t, typename offset_t>
  static FF_INLINE FF_DEVICE void 
  invert_(offset_t C, reduce_t * h, offset_t sh, reduce_t * v, offset_t sv,
          const reduce_t * w = static_cast<const reduce_t*>(nullptr),
          offset_t sw = static_cast<offset_t>(1)) {
    reduce_t * hh = h + C * sh;  // pointer to off-diagonal elements
    reduce_t oh = h[(C-1)*sh] + w[(C-1)*sw], ov = 0., tmp;
    for (offset_t c = 0; c < C-1; ++c, h += sh, hh += sh, w += sw, v += sv) {
      (*h) += (*w);
      tmp = (*hh) / (*h);
      oh -= (*hh) * tmp;
      ov += (*v) * tmp;
    }
    oh = 1. / oh; // oh = 1/mini_inv, ov = sum(vec_norm * grad)
    (*v) = tmp = ((*v) - ov) * oh;
    v -= sv; h -= sh; hh -= sh;
    for (offset_t c = 0; c < C-1; ++c, v -= sv, h -= sh, hh -=sh)
      (*v) = ((*v) - tmp * (*hh)) / (*h);
  }
};

template <>
struct HessianUtils<HessianType::Sym>: HessianCommonSym
{
  template <typename offset_t>
  static FF_INLINE offset_t work_size(offset_t C) { return C*C; }

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static FF_INLINE FF_DEVICE void 
  get(offset_t C, const scalar_t * i, offset_t si, reduce_t * o, offset_t so)
  {
    for (offset_t c = 0; c < C; ++c, i += si)
      o[(c+C*c)*so] = (*i) * OnePlusTiny;
    for (offset_t c = 0; c < C; ++c)
      for (offset_t cc = c+1; cc < C; ++cc, i += si)
        o[(c+C*cc)*so] = o[(cc+C*c)*so] = *i;
  }

  template <typename scalar_t, typename offset_t, typename reduce_t> 
  static FF_INLINE FF_DEVICE void 
  submatvec_(offset_t C, const scalar_t * i, offset_t si,
             const reduce_t * h, offset_t sh, reduce_t * o, offset_t so)
  {
    reduce_t acc;
    for (offset_t c = 0; c < C; ++c, o += so) {
      acc = static_cast<reduce_t>(0);
      for (offset_t cc = 0; cc < C; ++cc)
        acc += h[(c*C+cc)*sh] * i[cc*si];
      (*o) -= acc;
    }
  }

  template <typename reduce_t, typename offset_t>
  static FF_INLINE FF_DEVICE void invert_(
    offset_t C, reduce_t * h, offset_t sh, reduce_t * v, offset_t sv,
    const reduce_t * w = static_cast<const reduce_t*>(nullptr),
    offset_t sw = static_cast<offset_t>(1))
  {
    for (offset_t c = 0; c < C; ++c, w += sw)
      h[c*(C+1)*sh] += (*w);
    Cholesky::decompose(C, h, sh);      // cholesky decomposition
    Cholesky::solve(C, h, sh, v, sv);   // solve linear system inplace
  }
};


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                            Dispatcher
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#define FF_CASE_ENUM_USING_HINT(value, HINT, ...)             \
  case value: {                                               \
    static const auto HINT = value;                           \
    return __VA_ARGS__();                                     \
  }

#define FF_CASE_HESSIAN(type, ...) \
  FF_CASE_ENUM_USING_HINT(type, hessian_t, __VA_ARGS__)

#define FF_DFLT_ENUM_USING_HINT(value, HINT, ...)             \
  default: {                                                  \
    static const auto HINT = value;                           \
    return __VA_ARGS__();                                     \
  }

#define FF_DFLT_HESSIAN(type, ...) \
  FF_DFLT_ENUM_USING_HINT(type, hessian_t, __VA_ARGS__)

#define FF_DISPATCH_HESSIAN_TYPE(TYPE, ...)                                 \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    switch (the_type) {                                                     \
      FF_CASE_HESSIAN(HessianType::None,     __VA_ARGS__)                   \
      FF_CASE_HESSIAN(HessianType::Eye,      __VA_ARGS__)                   \
      FF_CASE_HESSIAN(HessianType::Diag,     __VA_ARGS__)                   \
      FF_CASE_HESSIAN(HessianType::ESTATICS, __VA_ARGS__)                   \
      FF_CASE_HESSIAN(HessianType::Sym,      __VA_ARGS__)                   \
    }                                                                       \
  }()

// If 1 channel (logC = 0) -> only None and Eye needed
#define FF_DISPATCH_HESSIAN_TYPE0(TYPE, ...)                                \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    switch (the_type) {                                                     \
      FF_CASE_HESSIAN(HessianType::None,     __VA_ARGS__)                   \
      FF_DFLT_HESSIAN(HessianType::Eye,      __VA_ARGS__)                   \
    }                                                                       \
  }()

// If 2 channels (logC = 1) -> only None, Eye and ESTATICS needed
#define FF_DISPATCH_HESSIAN_TYPE1(TYPE, ...)                                \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    switch (the_type) {                                                     \
      FF_CASE_HESSIAN(HessianType::None,     __VA_ARGS__)                   \
      FF_CASE_HESSIAN(HessianType::Eye,      __VA_ARGS__)                   \
      FF_DFLT_HESSIAN(HessianType::ESTATICS, __VA_ARGS__)                   \
    }                                                                       \
  }()


} // namespace device
} // namespace ff
