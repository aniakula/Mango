#pragma once

#include "shape.h"
#include "tensor.h"
#include "types.h"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

namespace mango::detail {

std::string dtype_name(DType type);

void assert_same_shape(const Shape &s1, const Shape &s2);
void assert_same_type(const DType &s1, const DType &s2);

void print_shape(std::ostream &os, const Shape &shape);

bool is_contiguous_layout(const Shape &shape, const Shape &strides);
size_t linear_index(size_t flat, const Shape &shape, const Shape &strides);
void copy_strided_to_contiguous(void *dst, const void *src, size_t n,
                                size_t elem_size, const Shape &shape,
                                const Shape &strides);

bool is_scalar_like(const Tensor &t);

enum class Reduction { Sum, Max, Min };

void dispatch_reduction(const void *input, void *output, size_t n, DType dtype,
                        Reduction reduction);
void divide_scalar_inplace(void *data, DType dtype, size_t divisor);

Tensor tensorMatMul(Tensor A, Tensor B);

template <typename F, typename T>
void apply_cast(Tensor &src, Tensor &new_tensor) {
  F *src_ptr = static_cast<F *>(src.data());
  T *to_ptr = static_cast<T *>(new_tensor.data());
  for (size_t idx = 0; idx < src.numel(); idx++) {
    to_ptr[idx] = static_cast<T>(*src_ptr);
  }
}

template <typename T> void tensor_cast(Tensor &src, Tensor &new_tensor) {
  switch (src.dtype()) {
  case DType::F32: {
    apply_cast<float, T>(src, new_tensor);
    break;
  }
  case DType::F64: {
    apply_cast<double, T>(src, new_tensor);
    break;
  }
  case DType::I32: {
    apply_cast<int32_t, T>(src, new_tensor);
    break;
  }
  case DType::I64: {
    apply_cast<int64_t, T>(src, new_tensor);
    break;
  }
  case DType::B: {
    apply_cast<bool, T>(src, new_tensor);
    break;
  }
  }
}

template <typename T, typename Op>
void apply_Op(void *lhs, const void *rhs, size_t n, Op op) {
  T *a = static_cast<T *>(lhs);
  const T *b = static_cast<const T *>(rhs);
  for (size_t i = 0; i < n; ++i) {
    a[i] = op(a[i], b[i]);
  }
}

template <typename T, typename Op>
void apply_Op_scalar(void *lhs, const void *scalar, size_t n, Op op) {
  T *a = static_cast<T *>(lhs);
  const T s = *static_cast<const T *>(scalar);
  for (size_t i = 0; i < n; ++i) {
    a[i] = op(a[i], s);
  }
}

template <typename Op>
void dispatch_Op(void *lhs, const void *rhs, size_t n, DType dtype, Op op) {
  switch (dtype) {
  case DType::F32:
    apply_Op<float>(lhs, rhs, n, op);
    break;
  case DType::F64:
    apply_Op<double>(lhs, rhs, n, op);
    break;
  case DType::I32:
    apply_Op<int32_t>(lhs, rhs, n, op);
    break;
  case DType::I64:
    apply_Op<int64_t>(lhs, rhs, n, op);
    break;
  case DType::B:
    throw std::invalid_argument("element-wise op does not support bool");
  }
}

template <typename Op>
void dispatch_Op_scalar(void *lhs, const void *scalar, size_t n, DType dtype,
                        Op op) {
  switch (dtype) {
  case DType::F32:
    apply_Op_scalar<float>(lhs, scalar, n, op);
    break;
  case DType::F64:
    apply_Op_scalar<double>(lhs, scalar, n, op);
    break;
  case DType::I32:
    apply_Op_scalar<int32_t>(lhs, scalar, n, op);
    break;
  case DType::I64:
    apply_Op_scalar<int64_t>(lhs, scalar, n, op);
    break;
  case DType::B:
    throw std::invalid_argument("element-wise op does not support bool");
  }
}

template <typename Op>
void elementwise_inplace(Tensor &lhs, const Tensor &rhs, Op op) {
  assert_same_type(lhs.dtype(), rhs.dtype());

  if (is_scalar_like(rhs)) {
    Tensor scalar = rhs.contiguous();
    dispatch_Op_scalar(lhs.data(), scalar.data(), lhs.numel(), lhs.dtype(), op);
    return;
  }

  if (is_scalar_like(lhs) && !is_scalar_like(rhs)) {
    throw std::invalid_argument("in-place op cannot broadcast a non-scalar "
                                "into a scalar left-hand side");
  }

  assert_same_shape(lhs.shape(), rhs.shape());
  Tensor other = rhs.is_contiguous() ? rhs : rhs.contiguous();
  dispatch_Op(lhs.data(), other.data(), lhs.numel(), lhs.dtype(), op);
}

template <typename Op>
Tensor elementwise_binary(const Tensor &lhs, const Tensor &rhs, Op op) {
  assert_same_type(lhs.dtype(), rhs.dtype());

  if (is_scalar_like(lhs) && !is_scalar_like(rhs)) {
    Tensor out = rhs.clone();
    Tensor scalar = lhs.contiguous();
    dispatch_Op_scalar(out.data(), scalar.data(), out.numel(), out.dtype(),
                       [&](auto a, auto s) { return op(s, a); });
    return out;
  }

  Tensor out = lhs.clone();
  elementwise_inplace(out, rhs, op);
  return out;
}

} // namespace mango::detail
