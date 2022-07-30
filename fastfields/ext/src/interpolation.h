#pragma once

// This file contains static functions for handling (0-7 order) 
// interpolation weights.
// It also defines an enumerated types that encodes each boundary type.
// The entry points are:
// . ff::interpolation::weight -> node weight based on distance
// . ff::interpolation::grad   -> weight derivative // oriented distance
// . ff::InterpolationType     -> enumerated type
//
// Everything in this file should have internal linkage (static) except
// the BoundType/BoundVectorRef types.

#include <ATen/ATen.h>
namespace ff {

enum class InterpolationType : int64_t
    {Nearest, Linear, Quadratic, Cubic, 
     FourthOrder, FifthOrder, SixthOrder, SeventhOrder};
using InterpolationVectorRef = c10::ArrayRef<InterpolationType>;

} // namespace ff