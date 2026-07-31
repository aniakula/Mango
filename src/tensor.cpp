#include "tensor.h"

#if defined(__APPLE__) && defined(__MACH__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#else
// 2. Fallback for Linux / Windows using standard OpenBLAS or MKL
#include <cblas.h>
#endif

#include <algorithm>
#include <cstring>
#include <format>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace mango {

namespace {

std::string dtype_name(DType type) {
  switch (type) {
  case DType::F32:
    return "f32";
  case DType::F64:
    return "f64";
  case DType::I32:
    return "i32";
  case DType::I64:
    return "i64";
  case DType::B:
    return "bool";
  }
  return "unknown";
}

void assert_same_shape(const Shape &s1, const Shape &s2) {
  if (s1.rank() != s2.rank()) {
    std::string error = std::format(
        "invalid shape ranks of {} and {}, must be equal for this operation",
        s1.rank(), s2.rank());
    throw std::invalid_argument(error);
  }

  for (size_t i = 0; i < s1.rank(); i++) {
    if (s1[i] != s2[i]) {
      throw std::invalid_argument(
          "invalid shape dimensions must be equal for this operation");
    }
  }
}

void assert_same_type(const DType &s1, const DType &s2) {
  if (s1 != s2) {
    std::string error = std::format(
        "invalid Tensor types of {} and {}, must be equal for this operation",
        dtype_name(s1), dtype_name(s2));
    throw std::invalid_argument(error);
  }
}

void print_shape(std::ostream &os, const Shape &shape) {
  os << '(';
  for (size_t i = 0; i < shape.rank(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << shape[i];
  }
  os << ')';
}

bool is_contiguous_layout(const Shape &shape, const Shape &strides) {
  return strides.to_vector() == Shape::compute_strides(shape).to_vector();
}

size_t linear_index(size_t flat, const Shape &shape, const Shape &strides) {
  size_t idx = 0;
  size_t rem = flat;
  for (int d = static_cast<int>(shape.rank()) - 1; d >= 0; --d) {
    const size_t dim = shape[static_cast<size_t>(d)];
    const size_t coord = dim == 0 ? 0 : rem % dim;
    rem = dim == 0 ? 0 : rem / dim;
    idx += coord * strides[static_cast<size_t>(d)];
  }
  return idx;
}

void copy_strided_to_contiguous(void *dst, const void *src, size_t n,
                                size_t elem_size, const Shape &shape,
                                const Shape &strides) {
  auto *out = static_cast<char *>(dst);
  const auto *in = static_cast<const char *>(src);
  for (size_t flat = 0; flat < n; ++flat) {
    const size_t src_idx = linear_index(flat, shape, strides);
    std::memcpy(out + flat * elem_size, in + src_idx * elem_size, elem_size);
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

// Contiguous element-wise op: lhs[i] = op(lhs[i], rhs[i]).
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

// Batched GEMM over leading dims. Last two dims are (M,K) @ (K,N).
// Supports identical batch shapes, or broadcasting a rank-2 operand.
Tensor tensorMatMul(Tensor A, Tensor B) {
  assert_same_type(A.dtype(), B.dtype());

  if (!A.is_contiguous()) {
    A = A.contiguous();
  }
  if (!B.is_contiguous()) {
    B = B.contiguous();
  }

  const size_t nDimsA = A.shape().rank();
  const size_t nDimsB = B.shape().rank();
  if (nDimsA < 2 || nDimsB < 2) {
    throw std::invalid_argument("matmul: tensors must be at least 2D");
  }

  const size_t M = A.shape()[nDimsA - 2];
  const size_t K = A.shape()[nDimsA - 1];
  const size_t K2 = B.shape()[nDimsB - 2];
  const size_t N = B.shape()[nDimsB - 1];
  if (K != K2) {
    throw std::invalid_argument("matmul: inner dimensions must match (K)");
  }

  size_t batchA = 1;
  for (size_t i = 0; i + 2 < nDimsA; ++i) {
    batchA *= A.shape()[i];
  }
  size_t batchB = 1;
  for (size_t i = 0; i + 2 < nDimsB; ++i) {
    batchB *= B.shape()[i];
  }

  std::vector<size_t> out_dims;
  if (nDimsA == 2 && nDimsB == 2) {
    out_dims = {M, N};
  } else if (nDimsA == 2) {
    // (M,K) @ (...,K,N) -> (...,M,N)
    out_dims = B.shape().to_vector();
    out_dims[out_dims.size() - 2] = M;
    out_dims[out_dims.size() - 1] = N;
  } else if (nDimsB == 2) {
    // (...,M,K) @ (K,N) -> (...,M,N)
    out_dims = A.shape().to_vector();
    out_dims[out_dims.size() - 2] = M;
    out_dims[out_dims.size() - 1] = N;
  } else {
    if (nDimsA != nDimsB) {
      throw std::invalid_argument(
          "matmul: batch ranks must match (or one operand is 2D)");
    }
    for (size_t i = 0; i + 2 < nDimsA; ++i) {
      if (A.shape()[i] != B.shape()[i]) {
        throw std::invalid_argument("matmul: batch dimensions must match");
      }
    }
    out_dims = A.shape().to_vector();
    out_dims[out_dims.size() - 2] = M;
    out_dims[out_dims.size() - 1] = N;
  }

  if (batchA != batchB && batchA != 1 && batchB != 1) {
    throw std::invalid_argument("matmul: incompatible batch sizes");
  }
  const size_t totalBatches = std::max(batchA, batchB);

  Tensor C = Tensor::zeros(Shape(out_dims), A.dtype());

  const size_t strideA = M * K;
  const size_t strideB = K * N;
  const size_t strideC = M * N;

  const int iM = static_cast<int>(M);
  const int iN = static_cast<int>(N);
  const int iK = static_cast<int>(K);

  switch (A.dtype()) {
  case DType::F32: {
    const float *a_base = static_cast<const float *>(A.data());
    const float *b_base = static_cast<const float *>(B.data());
    float *c_base = static_cast<float *>(C.data());
    for (size_t b = 0; b < totalBatches; ++b) {
      const float *sliceA = a_base + (batchA == 1 ? 0 : b) * strideA;
      const float *sliceB = b_base + (batchB == 1 ? 0 : b) * strideB;
      float *sliceC = c_base + b * strideC;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, iM, iN, iK, 1.0f,
                  sliceA, iK, sliceB, iN, 0.0f, sliceC, iN);
    }
    break;
  }
  case DType::F64: {
    const double *a_base = static_cast<const double *>(A.data());
    const double *b_base = static_cast<const double *>(B.data());
    double *c_base = static_cast<double *>(C.data());
    for (size_t b = 0; b < totalBatches; ++b) {
      const double *sliceA = a_base + (batchA == 1 ? 0 : b) * strideA;
      const double *sliceB = b_base + (batchB == 1 ? 0 : b) * strideB;
      double *sliceC = c_base + b * strideC;
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, iM, iN, iK, 1.0,
                  sliceA, iK, sliceB, iN, 0.0, sliceC, iN);
    }
    break;
  }
  default:
    throw std::invalid_argument("matmul: only f32 and f64 supported via BLAS");
  }

  return C;
}

} // namespace

Tensor::Tensor(Shape shape, DType type)
    : shape_(std::move(shape)), strides_(Shape::compute_strides(shape_)),
      offset_(0), dtype_(type),
      storage_(std::make_shared<Storage>(shape_.numel() * dtype_size(type))) {}

Shape Tensor::shape() const { return shape_; }

Shape Tensor::strides() const { return strides_; }

DType Tensor::dtype() const { return dtype_; }

size_t Tensor::numel() const { return shape_.numel(); }

size_t Tensor::storage_offset() const { return offset_; }

void *Tensor::data() {
  return static_cast<char *>(storage_->data()) + offset_ * dtype_size(dtype_);
}

const void *Tensor::data() const {
  return static_cast<const char *>(storage_->data()) +
         offset_ * dtype_size(dtype_);
}

void Tensor::reshape(const Shape &newShape) {
  if (newShape.numel() != shape_.numel()) {
    throw std::invalid_argument("reshape: numel must stay the same");
  }
  shape_ = newShape;
  strides_ = Shape::compute_strides(newShape);
}

Tensor Tensor::view(const Shape &newShape) const {
  if (newShape.numel() != shape_.numel()) {
    throw std::invalid_argument("view: numel must stay the same");
  }
  Tensor out = *this;
  out.shape_ = newShape;
  out.strides_ = Shape::compute_strides(newShape);
  return out;
}

Tensor Tensor::clone() const {
  Tensor out = *this;
  const size_t nbytes = storage_->bytes();
  out.storage_ = std::make_shared<Storage>(nbytes);
  if (nbytes > 0) {
    std::memcpy(out.storage_->data(), storage_->data(), nbytes);
  }
  return out;
}

Tensor Tensor::clone(const Shape &newShape) const {
  Tensor out = clone();
  out.reshape(newShape);
  return out;
}

bool Tensor::is_contiguous() const {
  return is_contiguous_layout(shape_, strides_);
}

Tensor Tensor::contiguous() const {
  const size_t elem_size = dtype_size(dtype_);
  const size_t n = numel();
  const size_t nbytes = n * elem_size;

  if (is_contiguous()) {
    if (offset_ == 0 && storage_->bytes() == nbytes) {
      return *this;
    }
    Tensor out(shape_, dtype_);
    if (nbytes > 0) {
      std::memcpy(out.data(), data(), nbytes);
    }
    return out;
  }

  Tensor out(shape_, dtype_);
  if (nbytes > 0) {
    copy_strided_to_contiguous(out.data(), data(), n, elem_size, shape_,
                               strides_);
  }
  return out;
}

Tensor Tensor::zeros(const Shape &shape, DType type) {
  Tensor tensor(shape, type);
  if (tensor.storage_->bytes() > 0) {
    std::memset(tensor.storage_->data(), 0, tensor.storage_->bytes());
  }
  return tensor;
}

void Tensor::transpose(size_t dim1, size_t dim2) {
  if (dim1 >= shape_.rank() || dim2 >= shape_.rank()) {
    throw std::out_of_range("transpose: dimension out of range");
  }
  auto dims = shape_.to_vector();
  auto str = strides_.to_vector();
  std::swap(dims[dim1], dims[dim2]);
  std::swap(str[dim1], str[dim2]);
  shape_ = Shape(dims);
  strides_ = Shape(str);
}

Tensor Tensor::operator+(const Tensor &other) const {
  Tensor out = clone();
  out += other;
  return out;
}

Tensor Tensor::operator-(const Tensor &other) const {
  Tensor out = clone();
  out -= other;
  return out;
}

Tensor Tensor::operator*(const Tensor &other) const {
  Tensor out = clone();
  out *= other;
  return out;
}

Tensor &Tensor::operator+=(const Tensor &other) {
  assert_same_shape(shape_, other.shape());
  assert_same_type(dtype_, other.dtype());
  dispatch_Op(data(), other.data(), numel(), dtype_,
              [](auto a, auto b) { return a + b; });
  return *this;
}

Tensor &Tensor::operator-=(const Tensor &other) {
  assert_same_shape(shape_, other.shape());
  assert_same_type(dtype_, other.dtype());
  dispatch_Op(data(), other.data(), numel(), dtype_,
              [](auto a, auto b) { return a - b; });
  return *this;
}

Tensor &Tensor::operator*=(const Tensor &other) {
  assert_same_shape(shape_, other.shape());
  assert_same_type(dtype_, other.dtype());
  dispatch_Op(data(), other.data(), numel(), dtype_,
              [](auto a, auto b) { return a * b; });
  return *this;
}

Tensor Tensor::matmul(const Tensor &other) const {
  return tensorMatMul(*this, other);
}

void Tensor::log(std::ostream &os) const {
  os << "Tensor dtype=" << dtype_name(dtype_) << " shape=";
  print_shape(os, shape_);
  os << " strides=";
  print_shape(os, strides_);
  os << " offset=" << offset_;
  os << " contiguous=" << (is_contiguous() ? "true" : "false");
  os << "\n";

  if (numel() == 0) {
    os << "  (empty)\n";
    return;
  }

  const size_t elem_size = dtype_size(dtype_);
  const auto *base =
      static_cast<const char *>(storage_->data()) + offset_ * elem_size;

  const auto print_row = [&](auto tag) {
    using T = decltype(tag);
    for (size_t flat = 0; flat < numel(); ++flat) {
      size_t storage_idx = flat;
      if (!is_contiguous()) {
        storage_idx = linear_index(flat, shape_, strides_);
      }
      const T value =
          *reinterpret_cast<const T *>(base + storage_idx * elem_size);
      if (flat > 0 && flat % shape_[shape_.rank() - 1] == 0) {
        os << "\n";
      }
      os << std::setw(10) << value << " ";
    }
    os << "\n";
  };

  os << "  data:\n  ";
  switch (dtype_) {
  case DType::F32:
    print_row(float{});
    break;
  case DType::F64:
    print_row(double{});
    break;
  case DType::I32:
    print_row(int32_t{});
    break;
  case DType::I64:
    print_row(int64_t{});
    break;
  case DType::B: {
    for (size_t flat = 0; flat < numel(); ++flat) {
      size_t storage_idx = flat;
      if (!is_contiguous()) {
        storage_idx = linear_index(flat, shape_, strides_);
      }
      const bool value = base[storage_idx] != 0;
      if (flat > 0 && flat % shape_[shape_.rank() - 1] == 0) {
        os << "\n  ";
      }
      os << std::setw(10) << (value ? "true" : "false") << " ";
    }
    os << "\n";
    break;
  }
  }
}

} // namespace mango
