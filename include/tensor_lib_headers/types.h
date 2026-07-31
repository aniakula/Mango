#pragma once
#include <coroutine>
#include <cstddef>
#include <type_traits>
#include <typeinfo>
namespace mango {
// centralized data types supported for tensor storage
enum class DType { F32, F64, I32, I64, B };
inline size_t dtype_size(DType dtype) {
  switch (dtype) {
  case DType::F32:
    return 4;
  case DType::F64:
    return 8;
  case DType::I32:
    return 4;
  case DType::I64:
    return 8;
  case DType::B:
    return 1;
  }
}

template <typename> inline constexpr bool always_false = false;

template <typename T> constexpr DType type_of() {
  if constexpr (std::is_same_v<T, float>) {
    return DType::F32;
  } else if constexpr (std::is_same_v<T, double>) {
    return DType::F64;

  } else if constexpr (std::is_same_v<T, int32_t>) {
    return DType::I32;

  } else if constexpr (std::is_same_v<T, int64_t>) {
    return DType::I64;
  } else if constexpr (std::is_same_v<T, bool>) {
    return DType::B;
  } else {
    static_assert(always_false<T>, "Unsupported type");
  }
}
// underlying device
enum class Device { CPU, CUDA };
} // namespace mango
