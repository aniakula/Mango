#include "tensor_lib_headers/tensor_internal.h"

#if defined(__APPLE__) && defined(__MACH__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <algorithm>
#include <cstring>
#include <format>
#include <stdexcept>
#include <vector>

namespace mango::detail {

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

bool is_scalar_like(const Tensor &t) { return t.numel() == 1; }

template <typename T>
T reduce_values(const T *values, size_t n, Reduction reduction) {
  if (reduction == Reduction::Sum) {
    T result{};
    for (size_t i = 0; i < n; ++i) {
      result += values[i];
    }
    return result;
  }

  if (n == 0) {
    throw std::invalid_argument("reduction: empty tensor has no max or min");
  }

  T result = values[0];
  for (size_t i = 1; i < n; ++i) {
    if (reduction == Reduction::Max) {
      result = std::max(result, values[i]);
    } else {
      result = std::min(result, values[i]);
    }
  }
  return result;
}

template <typename T>
void reduce_typed(const void *input, void *output, size_t n,
                  Reduction reduction) {
  const auto *values = static_cast<const T *>(input);
  *static_cast<T *>(output) = reduce_values(values, n, reduction);
}

void dispatch_reduction(const void *input, void *output, size_t n, DType dtype,
                        Reduction reduction) {
  switch (dtype) {
  case DType::F32:
    reduce_typed<float>(input, output, n, reduction);
    break;
  case DType::F64:
    reduce_typed<double>(input, output, n, reduction);
    break;
  case DType::I32:
    reduce_typed<int32_t>(input, output, n, reduction);
    break;
  case DType::I64:
    reduce_typed<int64_t>(input, output, n, reduction);
    break;
  case DType::B:
    throw std::invalid_argument("reduction: bool tensors are not supported");
  }
}

void divide_scalar_inplace(void *data, DType dtype, size_t divisor) {
  switch (dtype) {
  case DType::F32:
    *static_cast<float *>(data) /= static_cast<float>(divisor);
    break;
  case DType::F64:
    *static_cast<double *>(data) /= static_cast<double>(divisor);
    break;
  case DType::I32:
  case DType::I64:
    throw std::invalid_argument(
        "mean: only floating-point tensors are supported");
  case DType::B:
    throw std::invalid_argument("mean: bool tensors are not supported");
  }
}

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
    out_dims = B.shape().to_vector();
    out_dims[out_dims.size() - 2] = M;
    out_dims[out_dims.size() - 1] = N;
  } else if (nDimsB == 2) {
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

namespace {

bool is_float_dtype(DType dtype) {
  return dtype == DType::F32 || dtype == DType::F64;
}

#if defined(__clang__)
#define MANGO_LOOP_SCALAR                                                      \
  _Pragma("clang loop vectorize(disable) interleave(disable)")
#define MANGO_LOOP_VECTOR                                                      \
  _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__)
#define MANGO_LOOP_SCALAR _Pragma("GCC optimize(\"no-tree-vectorize\")")
#define MANGO_LOOP_VECTOR
#else
#define MANGO_LOOP_SCALAR
#define MANGO_LOOP_VECTOR
#endif

void check_poly_input(const Tensor &x, const std::vector<float> &coeffs) {
  if (x.dtype() != DType::F32) {
    throw std::invalid_argument("fused_poly: only F32 is supported");
  }
  if (coeffs.empty()) {
    throw std::invalid_argument("fused_poly: need at least one coefficient");
  }
}

Shape simd_out_shape(const Tensor &a, const Tensor &b) {
  if (is_scalar_like(a) && is_scalar_like(b)) {
    return a.shape();
  }
  if (is_scalar_like(a) && !is_scalar_like(b)) {
    return b.shape();
  }
  if (is_scalar_like(b) && !is_scalar_like(a)) {
    return a.shape();
  }
  assert_same_shape(a.shape(), b.shape());
  return a.shape();
}

} // namespace

Tensor fused_poly_loop(const Tensor &x_in, const std::vector<float> &coeffs) {
  check_poly_input(x_in, coeffs);
  const Tensor x = x_in.contiguous();
  Tensor out(x.shape(), x.dtype());

  const size_t n = x.numel();
  const size_t degree = coeffs.size() - 1;
  const float *in = static_cast<const float *>(x.data());
  float *dst = static_cast<float *>(out.data());
  const float *c = coeffs.data();

  MANGO_LOOP_SCALAR
  for (size_t i = 0; i < n; ++i) {
    const float xi = in[i];
    float acc = c[0];
    for (size_t k = 1; k <= degree; ++k) {
      acc = acc * xi + c[k]; // fused multiply-add
    }
    dst[i] = acc;
  }
  return out;
}

#if MANGO_ENABLE_VDSP

Tensor fused_poly_simd(const Tensor &x_in, const std::vector<float> &coeffs) {
  check_poly_input(x_in, coeffs);
  const Tensor x = x_in.contiguous();
  Tensor out(x.shape(), x.dtype());

  const vDSP_Length n = static_cast<vDSP_Length>(x.numel());
  const vDSP_Length degree = static_cast<vDSP_Length>(coeffs.size() - 1);
  vDSP_vpoly(coeffs.data(), 1, static_cast<const float *>(x.data()), 1,
             static_cast<float *>(out.data()), 1, n, degree);
  return out;
}

#else // !MANGO_ENABLE_VDSP

// No vDSP: same Horner loop, but let clang auto-vectorize it (many lanes).
Tensor fused_poly_simd(const Tensor &x_in, const std::vector<float> &coeffs) {
  check_poly_input(x_in, coeffs);
  const Tensor x = x_in.contiguous();
  Tensor out(x.shape(), x.dtype());

  const size_t n = x.numel();
  const size_t degree = coeffs.size() - 1;
  const float *in = static_cast<const float *>(x.data());
  float *dst = static_cast<float *>(out.data());
  const float *c = coeffs.data();

  MANGO_LOOP_VECTOR
  for (size_t i = 0; i < n; ++i) {
    const float xi = in[i];
    float acc = c[0];
    for (size_t k = 1; k <= degree; ++k) {
      acc = acc * xi + c[k];
    }
    dst[i] = acc;
  }
  return out;
}

#endif // MANGO_ENABLE_VDSP

#if MANGO_ENABLE_VDSP

namespace {

enum class SimdBinary { Add, Sub, Mul };

Tensor simd_binary_vdsp(const Tensor &a_in, const Tensor &b_in, SimdBinary op) {
  assert_same_type(a_in.dtype(), b_in.dtype());
  if (!is_float_dtype(a_in.dtype())) {
    throw std::invalid_argument(
        "simd binary: only floating-point dtypes are supported");
  }

  const Tensor a = a_in.contiguous();
  const Tensor b = b_in.contiguous();
  const Shape out_shape = simd_out_shape(a, b);
  Tensor result(out_shape, a.dtype());
  const vDSP_Length n = static_cast<vDSP_Length>(result.numel());

  const bool a_scalar = is_scalar_like(a);
  const bool b_scalar = is_scalar_like(b);
  // Two scalars (any rank-0 / {1} shapes): use the dense vector path with n=1.
  const bool use_scalar_rhs = b_scalar && !a_scalar;
  const bool use_scalar_lhs = a_scalar && !b_scalar;

  switch (a.dtype()) {
  case DType::F32: {
    const float *ap = static_cast<const float *>(a.data());
    const float *bp = static_cast<const float *>(b.data());
    float *out = static_cast<float *>(result.data());

    if (use_scalar_lhs) {
      const float s = *ap;
      switch (op) {
      case SimdBinary::Add:
        vDSP_vsadd(bp, 1, &s, out, 1, n);
        break;
      case SimdBinary::Sub: {
        // s - b
        vDSP_vneg(bp, 1, out, 1, n);
        vDSP_vsadd(out, 1, &s, out, 1, n);
        break;
      }
      case SimdBinary::Mul:
        vDSP_vsmul(bp, 1, &s, out, 1, n);
        break;
      }
    } else if (use_scalar_rhs) {
      const float s = *bp;
      switch (op) {
      case SimdBinary::Add:
        vDSP_vsadd(ap, 1, &s, out, 1, n);
        break;
      case SimdBinary::Sub: {
        const float neg = -s;
        vDSP_vsadd(ap, 1, &neg, out, 1, n);
        break;
      }
      case SimdBinary::Mul:
        vDSP_vsmul(ap, 1, &s, out, 1, n);
        break;
      }
    } else {
      switch (op) {
      case SimdBinary::Add:
        vDSP_vadd(ap, 1, bp, 1, out, 1, n);
        break;
      case SimdBinary::Sub:
        vDSP_vsub(bp, 1, ap, 1, out, 1, n); // out = a - b
        break;
      case SimdBinary::Mul:
        vDSP_vmul(ap, 1, bp, 1, out, 1, n);
        break;
      }
    }
    break;
  }
  case DType::F64: {
    const double *ap = static_cast<const double *>(a.data());
    const double *bp = static_cast<const double *>(b.data());
    double *out = static_cast<double *>(result.data());

    if (use_scalar_lhs) {
      const double s = *ap;
      switch (op) {
      case SimdBinary::Add:
        vDSP_vsaddD(bp, 1, &s, out, 1, n);
        break;
      case SimdBinary::Sub: {
        vDSP_vnegD(bp, 1, out, 1, n);
        vDSP_vsaddD(out, 1, &s, out, 1, n);
        break;
      }
      case SimdBinary::Mul:
        vDSP_vsmulD(bp, 1, &s, out, 1, n);
        break;
      }
    } else if (use_scalar_rhs) {
      const double s = *bp;
      switch (op) {
      case SimdBinary::Add:
        vDSP_vsaddD(ap, 1, &s, out, 1, n);
        break;
      case SimdBinary::Sub: {
        const double neg = -s;
        vDSP_vsaddD(ap, 1, &neg, out, 1, n);
        break;
      }
      case SimdBinary::Mul:
        vDSP_vsmulD(ap, 1, &s, out, 1, n);
        break;
      }
    } else {
      switch (op) {
      case SimdBinary::Add:
        vDSP_vaddD(ap, 1, bp, 1, out, 1, n);
        break;
      case SimdBinary::Sub:
        vDSP_vsubD(bp, 1, ap, 1, out, 1, n); // out = a - b
        break;
      case SimdBinary::Mul:
        vDSP_vmulD(ap, 1, bp, 1, out, 1, n);
        break;
      }
    }
    break;
  }
  default:
    throw std::invalid_argument(
        "simd binary: only floating-point dtypes are supported");
  }

  return result;
}

} // namespace

Tensor simd_add(const Tensor &a, const Tensor &b) {
  return simd_binary_vdsp(a, b, SimdBinary::Add);
}

Tensor simd_sub(const Tensor &a, const Tensor &b) {
  return simd_binary_vdsp(a, b, SimdBinary::Sub);
}

Tensor simd_mult(const Tensor &a, const Tensor &b) {
  return simd_binary_vdsp(a, b, SimdBinary::Mul);
}

Tensor simd_negate(const Tensor &a) {
  if (!is_float_dtype(a.dtype())) {
    throw std::invalid_argument(
        "simd_negate: only floating-point dtypes are supported");
  }
  const Tensor src = a.contiguous();
  Tensor result(src.shape(), src.dtype());
  const vDSP_Length n = static_cast<vDSP_Length>(src.numel());
  switch (src.dtype()) {
  case DType::F32:
    vDSP_vneg(static_cast<const float *>(src.data()), 1,
              static_cast<float *>(result.data()), 1, n);
    break;
  case DType::F64:
    vDSP_vnegD(static_cast<const double *>(src.data()), 1,
               static_cast<double *>(result.data()), 1, n);
    break;
  default:
    throw std::invalid_argument(
        "simd_negate: only floating-point dtypes are supported");
  }
  return result;
}

#else // !MANGO_ENABLE_VDSP

Tensor simd_add(const Tensor &a, const Tensor &b) {
  const Tensor a_c = a.contiguous();
  const Tensor b_c = b.contiguous();
  return elementwise_binary(a_c, b_c, [](auto x, auto y) { return x + y; });
}

Tensor simd_sub(const Tensor &a, const Tensor &b) {
  const Tensor a_c = a.contiguous();
  const Tensor b_c = b.contiguous();
  return elementwise_binary(a_c, b_c, [](auto x, auto y) { return x - y; });
}

Tensor simd_mult(const Tensor &a, const Tensor &b) {
  const Tensor a_c = a.contiguous();
  const Tensor b_c = b.contiguous();
  return elementwise_binary(a_c, b_c, [](auto x, auto y) { return x * y; });
}

Tensor simd_negate(const Tensor &a) {
  const Tensor src = a.contiguous();
  Tensor result(src.shape(), src.dtype());
  const size_t n = src.numel();
  switch (src.dtype()) {
  case DType::F32: {
    const float *in = static_cast<const float *>(src.data());
    float *out = static_cast<float *>(result.data());
    for (size_t i = 0; i < n; ++i) {
      out[i] = -in[i];
    }
    break;
  }
  case DType::F64: {
    const double *in = static_cast<const double *>(src.data());
    double *out = static_cast<double *>(result.data());
    for (size_t i = 0; i < n; ++i) {
      out[i] = -in[i];
    }
    break;
  }
  default:
    throw std::invalid_argument(
        "simd_negate: only floating-point dtypes are supported");
  }
  return result;
}

#endif // MANGO_ENABLE_VDSP

} // namespace mango::detail
