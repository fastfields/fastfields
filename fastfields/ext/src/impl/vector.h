#include "movable.h"
#include "../common.h"
#include <vector>

namespace ff {

  /*********************************************************************
   *
   *                          MOVABLE VECTOR
   *
  *********************************************************************/

  /// Dynamic array that can be moved to CUDA devices
  /// It explicitly copies data on construction.
  template <typename T>
  class Vector: public Movable<true> {
  public:
    using iterator = T*;
    using const_iterator = const T*;
    using size_type = size_t;
    using value_type = T;
    using reverse_iterator = std::reverse_iterator<iterator>;

    /// Allocate a given size
    Vector(size_t size=0):
      _data(size ? new T[size] : static_cast<T*>(nullptr))
    {}

    /// Allocate and fill
    Vector(size_t size, const T & value):
      _data(size ? new T[size] : static_cast<T*>(nullptr))
    {
      for (i=0; i < size; ++i)
        _data[i] = value;
    }

    /// Copy from a buffer (provided as address + size)
    template <typename U>
    Vector(const U * data, size_t size):
      _data(size ? new T[size] : static_cast<T*>(nullptr))
    {
      for (i=0; i < size; ++i)
        _data[i] = static_cast<T>(data[i]);
    }

    /// Copy from a buffer (given as a range)
    template <typename U>
    Vector(const U * begin, const U * end):
      _data(begin == end ? static_cast<T*>(nullptr) : new T[end - other])
    {
      for (size_t i = 0, auto x = begin; x < end; ++x, ++i)
        _data[i] = static_cast<T>(*x);
    }

    /// Copy from a C array.
    template <typename U, size_t size>
    Vector(const U (&data)[size]):
      _data(N ? new T[size] : static_cast<T*>(nullptr))
    {
      for (size_t i = 0, i < size; ++x, ++i)
        _data[i] = static_cast<T>(*x);
    }

    /// Fallback: assume iterable
    template <typename Iterable>
    Vector(const Iterable & other):
      _data(other.size() ? static_cast<T*>(nullptr) : new T[other.size()])
    {
      for (size_t i=0, auto x=other.begin(); x<other.end(); ++x, ++i)
        _data[i] = static_cast<T>(*x);
    }

    virtual ~Vector()
    {
      if (_data) delete _data;
    }

#ifdef __CUDACC__
    /// Allocate a copy on the device
    template <typename Stream>
    FF_HOST void ref_to_device(Stream stream) {
        _data = alloc_and_copy_to_device(_data, stream);
    }
#endif

    FF_INLINE constexpr T& operator[](size_t index) {
    return _data[index];
    }

    FF_INLINE constexpr const T& operator[](size_t index) const {
    return _data[index];
    }

    FF_INLINE constexpr iterator begin() {
    return _data;
    }
    
    FF_INLINE constexpr iterator begin() const {
    return _data;
    }
    
    FF_INLINE constexpr const_iterator cbegin() const {
    return _data;
    }
    
    FF_INLINE constexpr reverse_iterator rend() {
    return reverse_iterator(begin());
    }
    
    FF_INLINE constexpr reverse_iterator rend() const {
    return reverse_iterator(begin());
    }
    
    FF_INLINE constexpr const T* data() const {
    return _data;
    }
    
  protected:
    T * _data;

  };

  /// Like Vector, but explicitly store the vector size, which 
  /// allows iterators and additional helpers to be defined.
  template <typename T>
  class SizedVector: public Vector<T> {
  public:
  
    /// Allocate a given size
    SizedVector(size_t size=0): Vector(size), _size(size) {}

    /// Allocate and fill
    SizedVector(size_t size, const T & value): Vector(size, value), _size(size) {}

    /// Copy from a buffer (provided as address + size)
    SizedVector(const T * data, size_t size): Vector(size, value), _size(size) {}

    /// Copy from a buffer (given as a range)
    SizedVector(const T * begin, const T * end): Vector(begin, end), _size(end-begin) {}

    /// Copy from a C array.
    template <size_t size>
    SizedVector(const T (&data)[size]): Vector(data), _size(size) {}

    /// Fallback: assume iterable
    template <typename Iterable>
    SizedVector(const Iterable & other): Vector(other), _size(other.size()) {}

    FF_INLINE constexpr size_t empty() const { return _size == 0; }
    FF_INLINE constexpr size_t size() const { return _size; }

    FF_INLINE constexpr iterator end() {
    return _data + _size;
    }
    
    FF_INLINE constexpr iterator end() const {
    return _data + _size;
    }
    
    FF_INLINE constexpr const_iterator cend() const {
    return _data + _size;
    }

  protected:
    size_t _size;
  };

#if 0
  /*********************************************************************
   *
   *                          VECTOR REF
   *
  *********************************************************************/

  /// Reference to an array that we do not own.
  template <typename T>
  class VectorRef {
  protected:
    T * _data;
    size_t _size;
  public:
    using iterator = T*;
    using const_iterator = const T*;
    using size_type = size_t;
    using value_type = T;
    using reverse_iterator = std::reverse_iterator<iterator>;

    /// Empty
    VectorRef(): _data(static_cast<T*>(nullptr)), _size(0) {}

    /// From an element
    VectorRef(T & data): _data(&data), _size(1) {}

    /// From a buffer (provided as address + size)
    VectorRef(T * data, size_t size):
      _data(size ? data : static_cast<T*>(nullptr)), _size(size) {}

    /// Copy from a buffer (given as a range)
    VectorRef(T * begin, T * end):
      _data(begin == end ? static_cast<T*>(nullptr) : begin), _size(end-begin)
     {}

    /// Copy from a C array.
    template <typename U, size_t size>
    VectorRef(U (&data)[size]):
      _data(N ? data : static_cast<T*>(nullptr)), _size(size) {}

    /// From a vector
    template <typename T>
    VectorRef(std::vector<T> & other):
      _data(other.data()), _size(other.size()) {}

    FF_INLINE constexpr T* data() { return _data; }
    FF_INLINE constexpr const T* data() const { return _data; }

    FF_INLINE constexpr size_t empty() const { return _size == 0; }
    FF_INLINE constexpr size_t size() const { return _size; }

    FF_INLINE constexpr T& operator[](size_t index) {
    return _data[index];
    }

    FF_INLINE constexpr const T& operator[](size_t index) const {
    return _data[index];
    }

    FF_INLINE constexpr iterator begin() {
    return _data;
    }

    FF_INLINE constexpr iterator end() {
    return _data + _size;
    }

    FF_INLINE constexpr iterator begin() const {
    return _data;
    }

    FF_INLINE constexpr iterator end() const {
    return _data + _size;
    }

    FF_INLINE constexpr const_iterator cbegin() const {
    return _data;
    }

    FF_INLINE constexpr const_iterator cend() const {
    return _data + _size;
    }

    FF_INLINE constexpr reverse_iterator rbegin() {
    return reverse_iterator(end());
    }

    FF_INLINE constexpr reverse_iterator rend() {
    return reverse_iterator(begin());
    }

    FF_INLINE constexpr reverse_iterator rbegin() const {
    return reverse_iterator(end());
    }

    FF_INLINE constexpr reverse_iterator rend() const {
    return reverse_iterator(begin());
    }

  };

  /// Reference to a constant array that we do not own. (like at::ArrayRef)
  template <typename T>
  class ConstVectorRef {
  protected:
    const T * _data;
    size_t _size;
  public:
    using iterator = const T*;
    using const_iterator = const T*;
    using size_type = size_t;
    using value_type = T;
    using reverse_iterator = std::reverse_iterator<iterator>;

    /// Empty
    ConstVectorRef(): _data(static_cast<T*>(nullptr)), _size(0) {}

    /// From an element
    ConstVectorRef(const T & data): _data(&data), _size(1) {}

    /// From a buffer (provided as address + size)
    ConstVectorRef(const T * data, size_t size):
      _data(size ? data : static_cast<T*>(nullptr)), _size(size) {}

    /// Copy from a buffer (given as a range)
    ConstVectorRef(const T * begin, const T * end):
      _data(begin == end ? static_cast<T*>(nullptr) : begin), _size(end-begin)
     {}

    /// Copy from a C array.
    template <typename U, size_t size>
    ConstVectorRef(const U (&data)[size]):
      _data(N ? data : static_cast<T*>(nullptr)), _size(size) {}

    /// From a vector
    template <typename T>
    ConstVectorRef(const std::vector<T> & other):
      _data(other.data()), _size(other.size()) {}

    FF_INLINE constexpr const T* data() const { return _data; }

    FF_INLINE constexpr size_t empty() const { return _size == 0; }
    FF_INLINE constexpr size_t size() const { return _size; }

    FF_INLINE constexpr const T& operator[](size_t index) const {
    return _data[index];
    }

    FF_INLINE constexpr iterator begin() const {
    return _data;
    }

    FF_INLINE constexpr iterator end() const {
    return _data + _size;
    }

    FF_INLINE constexpr const_iterator cbegin() const {
    return _data;
    }

    FF_INLINE constexpr const_iterator cend() const {
    return _data + _size;
    }

    FF_INLINE constexpr reverse_iterator rbegin() const {
    return reverse_iterator(end());
    }

    FF_INLINE constexpr reverse_iterator rend() const {
    return reverse_iterator(begin());
    }

  };


  /*********************************************************************
   *
   *                        STRIDED VECTOR REF
   *
  *********************************************************************/

template <typename T>
class StridedIterator {
protected:
  T * _data;
  size_t _stride;
public:
  strided_iterator(T * data, size_t stride)
  strided_iterator & operator++() {
    _data += stride;
    return *this;
  }
  strided_iterator & operator++(int) {
    previous = *this;
    _data += stride;
    return previous;
  }
  strided_iterator & operator--() {
    _data -= stride;
    return *this;
  }
  strided_iterator & operator--(int) {
    previous = *this;
    _data -= stride;
    return previous;
  }
  bool operator<(const strided_iterator & other) const {
    return _data < other._data;
  }
  bool operator<=(const strided_iterator & other) const {
    return _data <= other._data;
  }
  bool operator>(const strided_iterator & other) const {
    return _data > other._data;
  }
  bool operator>=(const strided_iterator & other) const {
    return _data >= other._data;
  }
  bool operator==(const strided_iterator & other) const {
    return _data == other._data;
  }
  bool operator!=(const strided_iterator & other) const {
    return _data != other._data;
  }
};

template <typename T>
class ConstStridedIterator {
protected:
  const T * _data;
  size_t _stride;
public:
  strided_iterator(const T * data, size_t stride)
  strided_iterator & operator++() {
    _data += stride;
    return *this;
  }
  strided_iterator & operator++(int) {
    previous = *this;
    _data += stride;
    return previous;
  }
  strided_iterator & operator--() {
    _data -= stride;
    return *this;
  }
  strided_iterator & operator--(int) {
    previous = *this;
    _data -= stride;
    return previous;
  }
  bool operator<(const strided_iterator & other) const {
    return _data < other._data;
  }
  bool operator<=(const strided_iterator & other) const {
    return _data <= other._data;
  }
  bool operator>(const strided_iterator & other) const {
    return _data > other._data;
  }
  bool operator>=(const strided_iterator & other) const {
    return _data >= other._data;
  }
  bool operator==(const strided_iterator & other) const {
    return _data == other._data;
  }
  bool operator!=(const strided_iterator & other) const {
    return _data != other._data;
  }
};

  /// Reference to a strided array that we do not own.
  template <typename T>
  class StridedVectorRef: public VectorRef<T> {
  protected:
    T * _data;
    size_t _size;
    size_t _stride;

  public:

    using iterator = StridedIterator<T>;
    using const_iterator = ConstStridedIterator<T>;
    using reverse_iterator = std::reverse_iterator<iterator>;

    /// Empty
    StridedVectorRef(): VectorRef(), _stride(1) {}

    /// From an element
    StridedVectorRef(T & data): VectorRef(data), _stride(1) {}

    /// From a buffer (provided as address + size)
    StridedVectorRef(T * data, size_t size, size_t stride=1):
      VectorRef(data, size), _stride(stride) {}

    /// Copy from a buffer (given as a range)
    StridedVectorRef(T * begin, T * end, size_t stride=1):
      VectorRef(begin, end), _size((begin-end)/stride), _stride(stride) {}

    /// Copy from a C array.
    template <typename U, size_t size>
    StridedVectorRef(U (&data)[size], size_t stride=1):
      VectorRef(data), _stride(stride) {}

    /// From a vector
    template <typename T>
    StridedVectorRef(std::vector<T> & other): VectorRef(other), _stride(1) {}

    FF_INLINE constexpr size_t stride() const { return _stride; }

    FF_INLINE constexpr T& operator[](size_t index) {
      return _data[index*_stride];
    }

    FF_INLINE constexpr const T& operator[](size_t index) const {
    return _data[index*_stride];
    }

    FF_INLINE constexpr iterator begin() {
    return iterator(_data, _stride);
    }

    FF_INLINE constexpr iterator end() {
    return iterator(_data + _size, _stride);
    }

    FF_INLINE constexpr const_iterator begin() const {
    return const_iterator(_data, _stride);
    }

    FF_INLINE constexpr const_iterator end() const {
    return const_iterator(_data + _size);
    }

    FF_INLINE constexpr const_iterator cbegin() const {
    return const_iterator(_data, _stride);
    }

    FF_INLINE constexpr const_iterator cend() const {
    return const_iterator(_data + _size, _stride);
    }

    FF_INLINE constexpr reverse_iterator rbegin() {
    return reverse_iterator(end());
    }

    FF_INLINE constexpr reverse_iterator rend() {
    return reverse_iterator(begin());
    }

    FF_INLINE constexpr reverse_iterator rbegin() const {
    return reverse_iterator(end());
    }

    FF_INLINE constexpr reverse_iterator rend() const {
    return reverse_iterator(begin());
    }
  }
#endif
}