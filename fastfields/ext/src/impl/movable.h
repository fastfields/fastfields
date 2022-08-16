#pragma once
#include "common.h"

namespace ff {

  /// Base class for objects that can be moved to CUDA devices
  /// TODO: from_device
  template <typename TypeToMove, bool HAS_REF = false>
  class Moveable {
  public:

#ifdef __CUDACC__
    template <typename Stream>
    FF_HOST TypeToMove * to_device(Stream stream) const {
        auto ptr_to_move = static_cast<const TypeToMove*>(this);
        return alloc_and_copy_to_device(ptr_to_move, stream);
    }
#else
    template <typename Stream>
    FF_HOST const TypeToMove * to_device(Stream stream) const {
        return reinterpret_cast<const TypeToMove*>(this);
    }

    template <typename Stream>
    FF_HOST TypeToMove * to_device(Stream stream) {
        return reinterpret_cast<TypeToMove*>(this);
    }
#endif

  };

  /// Objects that use pointers to the heap must be copied recursively.
  /// The easiest thing is to make a simple copy of the object (hoping
  /// that we do not trigger large copies on the heap) and mutate its
  /// fields before copying it to the device.
  template <typename TypeToMove>
  class Moveable<TypeToMove, true> {
  public:

    /// Move all pointed objects to device
    template <typename Stream>
    FF_HOST void ref_to_device(Stream stream) {}

#ifdef __CUDACC__
    template <typename Stream>
    FF_HOST TypeToMove * to_device(Stream stream) const {
        auto ptr_to_move = static_cast<const TypeToMove*>(this);
        TypeToMove copy = *ptr_to_move;
        copy.ref_to_device(stream);
        return alloc_and_copy_to_device(&copy, stream);
    }
#else
    template <typename Stream>
    FF_HOST const TypeToMove * to_device(Stream stream) const {
        return reinterpret_cast<const TypeToMove*>(this);
    }

    template <typename Stream>
    FF_HOST TypeToMove * to_device(Stream stream) {
        return reinterpret_cast<TypeToMove*>(this);
    }
#endif

  };

}
