#ifndef RAJA_REDUCTION_UTIL
#define RAJA_REDUCTION_UTIL

#include <functional>
#include <climits>
#include <cfloat>

#include <RAJA.hxx>

// TODO - best way to determine if targeting a GPU?
#ifdef USE_GPU
#define RAJA_HOST_DEVICE __host__ __device__
#else
#define RAJA_HOST_DEVICE
#endif

namespace RAJA {
namespace internal {
namespace Reduction {
namespace detail {

  template <typename T>
  struct limits {
    static constexpr RAJA_HOST_DEVICE const T zero() { return static_cast<T>(0); }
    static constexpr RAJA_HOST_DEVICE const T one()  { return static_cast<T>(1); }
  };

  template <>
  struct limits<int> {
    static constexpr RAJA_HOST_DEVICE const int max() { return 0x7FFFFFFF; }
    static constexpr RAJA_HOST_DEVICE const int min() { return 0xFFFFFFFF; }
  };

  template <>
  struct limits<unsigned int> {
    static constexpr RAJA_HOST_DEVICE const unsigned int max() { return 0xFFFFFFFF; }
    static constexpr RAJA_HOST_DEVICE const unsigned int min() { return 0x00000000; }
  };

  template <>
  struct limits<unsigned long int> {
    static constexpr RAJA_HOST_DEVICE const unsigned long int max() { return 0xFFFFFFFF; }
    static constexpr RAJA_HOST_DEVICE const unsigned long int min() { return 0x00000000; }
  };

  template <>
  struct limits<long long> {
    static constexpr RAJA_HOST_DEVICE const long long max() { return 0x7FFFFFFFFFFFFFFF; }
    static constexpr RAJA_HOST_DEVICE const long long min() { return 0xFFFFFFFFFFFFFFFF; }
  };

  template <>
  struct limits<unsigned long long> {
    static constexpr RAJA_HOST_DEVICE const unsigned long long max() { return 0xFFFFFFFFFFFFFFFF; }
    static constexpr RAJA_HOST_DEVICE const unsigned long long min() { return 0x0000000000000000; }
  };

  template <>
  struct limits<float> {
    static constexpr RAJA_HOST_DEVICE const float max() { return FLT_MAX; }
    static constexpr RAJA_HOST_DEVICE const float min() { return -FLT_MAX; }
  };

  template <>
  struct limits<double> {
    static constexpr RAJA_HOST_DEVICE const double max() { return DBL_MAX; }
    static constexpr RAJA_HOST_DEVICE const double min() { return -DBL_MAX; }
  };

  template <>
  struct limits<long double> {
    static constexpr RAJA_HOST_DEVICE const long double max() { return LDBL_MAX; }
    static constexpr RAJA_HOST_DEVICE const long double min() { return -LDBL_MAX; }
  };

} // detail

  template <typename T>
  struct Min {
    using type = T;
    using reduction_type = T;
    namespace detail = RAJA::internal::Reduction::detail;

    static inline
    constexpr RAJA_HOST_DEVICE
    const type
    init () {
      return detail::limits<T>::max();
    }

    static inline
    constexpr RAJA_HOST_DEVICE
    const reduction_type
    apply (const reduction_type & a, const reduction_type & b) noexcept {
      return (a <= b) ? a : b;
    }
  };

  template <typename T>
  struct Max {
    using type = T;
    using reduction_type = T;
    namespace detail = RAJA::internal::Reduction::detail;

    static inline
    constexpr RAJA_HOST_DEVICE
    const type
    init () {
      return detail::limits<T>::min();
    }

    static inline
    constexpr RAJA_HOST_DEVICE
    const reduction_type
    apply (const reduction_type & a, const reduction_type & b) noexcept {
      return (a >= b) ? a : b;
    }
  };

  template <typename T>
  struct Sum {
    using type = T;
    using reduction_type = T;
    namespace detail = RAJA::internal::Reduction::detail;

    static inline
    constexpr RAJA_HOST_DEVICE
    const type
    init () {
      return detail::limits<T>::zero();
    }

    static inline
    constexpr RAJA_HOST_DEVICE
    const reduction_type
    apply (const reduction_type & a, const reduction_type & b) noexcept {
      return a + b;
    }
  };

  template <typename T>
  struct Product {
    using type = T;
    using reduction_type = T;
    namespace detail = RAJA::internal::Reduction::detail;

    static inline
    constexpr RAJA_HOST_DEVICE
    const type
    init () {
      return detail::limits<T>::one();
    }

    static inline
    constexpr RAJA_HOST_DEVICE
    const reduction_type
    apply (const reduction_type & a, const reduction_type & b) noexcept {
      return a * b;
    }
  };

  template <typename T, typename IndexT = RAJA::Index_type>
  struct IndexValuePair {
    IndexT index;
    T value;

    constexpr RAJA_HOST_DEVICE
    IndexValuePair (IndexT i, const T& v) noexcept
      : index (i), value (v) { }

    constexpr RAJA_HOST_DEVICE
    IndexValuePair (IndexT && i, T && v) noexcept
      : index (i), value (v) { }

  };

  template <typename T, typename IndexT = RAJA::Index_type>
  struct MinLoc {
    using type = IndexValuePair<T, IndexT>;
    using reduction_type = IndexValuePair<T, IndexT>;
    namespace detail = RAJA::internal::Reduction::detail;

    static inline
    constexpr RAJA_HOST_DEVICE
    const type
    init () {
      return { detail::limits<IndexT>::max(), detail::limits<T>::max() };
    }

    static inline
    constexpr RAJA_HOST_DEVICE
    const reduction_type &
    apply (const reduction_type & a, const reduction_type & b) noexcept {
      return (a.value <= b.value) ? a : b;
    }
  };

  template <typename T, typename IndexT = RAJA::Index_type>
  struct MaxLoc {
    using type = IndexValuePair<T, IndexT>;
    using reduction_type = IndexValuePair<T, IndexT>;
    namespace detail = RAJA::internal::Reduction::detail;

    static inline
    constexpr RAJA_HOST_DEVICE
    const type
    init () {
      return { detail::limits<IndexT>::max(), detail::limits<T>::min() };
    }

    static inline
    constexpr RAJA_HOST_DEVICE
    const reduction_type &
    apply (const reduction_type & a, const reduction_type & b) noexcept {
      return (a.value >= b.value) ? a : b;
    }
  };

} // Reduction
} // internal
} // RAJA

#endif
