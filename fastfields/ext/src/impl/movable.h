#include "../common.h"

namespace ff {

  /// Base class for objects that can be moved to CUDA devices
  /// TODO: from_device
  template <bool HAS_REF = false>
  class Moveable {
  public:

#ifdef __CUDACC__
    template <typename Stream>
    FF_HOST Moveable * to_device(Stream stream) const {

        return alloc_and_copy_to_device(this, stream);
    }
#else
    template <typename Stream>
    FF_HOST Moveable * to_device(Stream stream) const {
        return this;
    }
#endif

  };

  /// Objects that use pointers to the heap must be copied recursively.
  /// The easiest thing is to make a simple copy of the object (hoping
  /// that we do not trigger large copies on the heap) and mutate its
  /// fields before copying it to the device.
  template <>
  class Moveable<true> {
  public:

    /// Move all pointed objects to device
    template <typename Stream>
    FF_HOST void ref_to_device(Stream stream) {}

#ifdef __CUDACC__
    template <typename Stream>
    FF_HOST Moveable * to_device(Stream stream) const {
        Moveable copy = *this;
        copy.ref_to_device()
        return alloc_and_copy_to_device(&copy, stream);
    }
#else
    template <typename Stream>
    FF_HOST Moveable * to_device(Stream stream) const {
        return this;
    }
#endif

  };

}