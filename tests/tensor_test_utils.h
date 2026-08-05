#pragma once

#include "tensor_lib_headers/tensor.h"

#include <gtest/gtest.h>

#include <vector>

namespace mango::test {

inline void expect_shape(const Tensor &tensor,
                         std::initializer_list<size_t> expected) {
  EXPECT_EQ(tensor.shape().to_vector(), std::vector<size_t>(expected));
}

template <typename T> std::vector<T> values(const Tensor &tensor) {
  Tensor contiguous = tensor.contiguous();
  const auto *data = static_cast<const T *>(contiguous.data());
  return std::vector<T>(data, data + contiguous.numel());
}

template <typename T>
void expect_values(const Tensor &tensor, std::initializer_list<T> expected) {
  EXPECT_EQ(values<T>(tensor), std::vector<T>(expected));
}

template <typename T>
void expect_values_near(const Tensor &tensor, std::initializer_list<T> expected,
                        double tolerance) {
  const std::vector<T> actual = values<T>(tensor);
  ASSERT_EQ(actual.size(), expected.size());
  size_t index = 0;
  for (const T value : expected) {
    EXPECT_NEAR(actual[index], value, tolerance) << "at index " << index;
    ++index;
  }
}

template <typename T>
void expect_grad_near(Tensor &tensor, std::initializer_list<T> expected,
                      double tolerance = 1e-6) {
  ASSERT_NE(tensor.grad(), nullptr);
  expect_values_near<T>(*tensor.grad(), expected, tolerance);
}

} // namespace mango::test
