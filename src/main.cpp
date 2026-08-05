#include "tensor_lib_headers/shape.h"
#include "tensor_lib_headers/tensor.h"
#include "tensor_lib_headers/tensor_internal.h"
#include "tensor_lib_headers/types.h"

#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using mango::DType;
using mango::Shape;
using mango::Tensor;
namespace detail = mango::detail;

namespace {

using Clock = std::chrono::steady_clock;

Tensor make_f32(std::initializer_list<size_t> dims, float fill) {
  Tensor t = Tensor::empty<float>(Shape(std::vector<size_t>(dims)));
  float *p = static_cast<float *>(t.data());
  for (size_t i = 0; i < t.numel(); ++i) {
    p[i] = fill + static_cast<float>(i % 97) * 0.01f;
  }
  return t;
}

double time_ms(const std::function<void()> &fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  const auto start = Clock::now();
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  const auto end = Clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count() /
         static_cast<double>(iters);
}

void print_row(const std::string &name, double scalar_ms, double simd_ms) {
  const double speedup = scalar_ms / std::max(simd_ms, 1e-12);
  std::cout << std::left << std::setw(28) << name << std::right << std::fixed
            << std::setprecision(3) << std::setw(12) << scalar_ms
            << std::setw(12) << simd_ms << std::setw(10) << std::setprecision(2)
            << speedup << "x\n";
}

template <typename ScalarOp, typename SimdFn>
void run_binary_bench(const std::string &name, const Tensor &a, const Tensor &b,
                      ScalarOp scalar_op, SimdFn simd_fn, int warmup,
                      int iters) {
  volatile float sink = 0.f;

  const double scalar_ms = time_ms(
      [&] {
        Tensor out = detail::elementwise_binary(a, b, scalar_op);
        sink += *static_cast<const float *>(out.data());
      },
      warmup, iters);

  const double simd_ms = time_ms(
      [&] {
        Tensor out = simd_fn(a, b);
        sink += *static_cast<const float *>(out.data());
      },
      warmup, iters);

  print_row(name, scalar_ms, simd_ms);
  (void)sink;
}

template <typename ScalarFn, typename SimdFn>
void run_unary_bench(const std::string &name, const Tensor &a,
                     ScalarFn scalar_fn, SimdFn simd_fn, int warmup,
                     int iters) {
  volatile float sink = 0.f;

  const double scalar_ms = time_ms(
      [&] {
        Tensor out = scalar_fn(a);
        sink += *static_cast<const float *>(out.data());
      },
      warmup, iters);

  const double simd_ms = time_ms(
      [&] {
        Tensor out = simd_fn(a);
        sink += *static_cast<const float *>(out.data());
      },
      warmup, iters);

  print_row(name, scalar_ms, simd_ms);
  (void)sink;
}

void print_fused_row(const std::string &name, double loop_ms, double simd_ms,
                     double intensity) {
  const double speedup = loop_ms / std::max(simd_ms, 1e-12);
  std::cout << std::left << std::setw(20) << name << std::right << std::fixed
            << std::setprecision(3) << std::setw(12) << loop_ms << std::setw(12)
            << simd_ms << std::setw(10) << std::setprecision(2) << speedup
            << "x" << std::setw(12) << std::setprecision(2) << intensity
            << "\n";
}

// Evaluate a degree-`degree` polynomial per element. Memory traffic is fixed
// (read x, write out) no matter the degree; only the FMA count grows. This is
// the single knob that slides the kernel from memory-bound to compute-bound.
void run_fused_bench(const Tensor &x, size_t degree, int warmup, int iters) {
  std::vector<float> coeffs(degree + 1);
  for (size_t k = 0; k < coeffs.size(); ++k) {
    coeffs[k] = 0.5f + static_cast<float>(k) * 0.01f;
  }

  volatile float sink = 0.f;
  const double loop_ms = time_ms(
      [&] {
        Tensor out = detail::fused_poly_loop(x, coeffs);
        sink += *static_cast<const float *>(out.data());
      },
      warmup, iters);
  const double simd_ms = time_ms(
      [&] {
        Tensor out = detail::fused_poly_simd(x, coeffs);
        sink += *static_cast<const float *>(out.data());
      },
      warmup, iters);
  (void)sink;

  // 2 FLOPs per FMA * degree FMAs, over 8 bytes moved (read+write one f32).
  const double intensity = static_cast<double>(2 * degree) / 8.0;
  print_fused_row("poly deg " + std::to_string(degree), loop_ms, simd_ms,
                  intensity);
}

Tensor scalar_negate(const Tensor &a) {
  Tensor src = a.contiguous();
  Tensor out(src.shape(), src.dtype());
  const float *in = static_cast<const float *>(src.data());
  float *dest = static_cast<float *>(out.data());
  for (size_t i = 0; i < src.numel(); ++i) {
    dest[i] = -in[i];
  }
  return out;
}

} // namespace

int main() {
  constexpr size_t n = 4 * 1024 * 1024;
  constexpr int warmup = 5;
  constexpr int iters = 30;

  std::cout << "Mango SIMD benchmark\n";
  std::cout << "  MANGO_ENABLE_VDSP compile flag: "
            << (detail::vdsp_enabled() ? "ON" : "OFF") << "\n";
  std::cout << "  elements: " << n << " (F32), warmup=" << warmup
            << ", iters=" << iters << "\n";
  if (!detail::vdsp_enabled()) {
    std::cout << "  note: rebuild with -DMANGO_ENABLE_VDSP=ON for a real "
                 "vDSP vs scalar comparison; simd_* falls back to scalar when "
                 "OFF.\n";
  }
  std::cout << "\n";

  Tensor a = make_f32({n}, 1.0f);
  Tensor b = make_f32({n}, 2.0f);
  Tensor scalar({3.5f}, Shape{});

  std::cout << std::left << std::setw(28) << "operation" << std::right
            << std::setw(12) << "scalar ms" << std::setw(12) << "simd ms"
            << std::setw(10) << "speedup\n";
  std::cout << std::string(62, '-') << "\n";

  // Same-shape vector ops
  run_binary_bench(
      "add (vec + vec)", a, b, [](auto x, auto y) { return x + y; },
      [](const Tensor &x, const Tensor &y) { return detail::simd_add(x, y); },
      warmup, iters);
  run_binary_bench(
      "sub (vec - vec)", a, b, [](auto x, auto y) { return x - y; },
      [](const Tensor &x, const Tensor &y) { return detail::simd_sub(x, y); },
      warmup, iters);
  run_binary_bench(
      "mul (vec * vec)", a, b, [](auto x, auto y) { return x * y; },
      [](const Tensor &x, const Tensor &y) { return detail::simd_mult(x, y); },
      warmup, iters);

  // Scalar ↔ vector broadcast
  run_binary_bench(
      "add (vec + scalar)", a, scalar, [](auto x, auto y) { return x + y; },
      [](const Tensor &x, const Tensor &y) { return detail::simd_add(x, y); },
      warmup, iters);
  run_binary_bench(
      "add (scalar + vec)", scalar, a, [](auto x, auto y) { return x + y; },
      [](const Tensor &x, const Tensor &y) { return detail::simd_add(x, y); },
      warmup, iters);
  run_binary_bench(
      "sub (vec - scalar)", a, scalar, [](auto x, auto y) { return x - y; },
      [](const Tensor &x, const Tensor &y) { return detail::simd_sub(x, y); },
      warmup, iters);
  run_binary_bench(
      "sub (scalar - vec)", scalar, a, [](auto x, auto y) { return x - y; },
      [](const Tensor &x, const Tensor &y) { return detail::simd_sub(x, y); },
      warmup, iters);
  run_binary_bench(
      "mul (vec * scalar)", a, scalar, [](auto x, auto y) { return x * y; },
      [](const Tensor &x, const Tensor &y) { return detail::simd_mult(x, y); },
      warmup, iters);
  run_binary_bench(
      "mul (scalar * vec)", scalar, a, [](auto x, auto y) { return x * y; },
      [](const Tensor &x, const Tensor &y) { return detail::simd_mult(x, y); },
      warmup, iters);

  // Unary
  run_unary_bench(
      "negate (-vec)", a, [](const Tensor &x) { return scalar_negate(x); },
      [](const Tensor &x) { return detail::simd_negate(x); }, warmup, iters);

  // Fused polynomial: same memory traffic, growing compute.
  // Low degree is memory-bound (SIMD ~= loop); high degree is compute-bound
  // (SIMD lanes win). This is the memory-vs-compute crossover.
  std::cout << "\nfused polynomial (Horner, single pass over memory)\n";
  std::cout << "  memory traffic fixed at 8 B/elem\n\n";
  std::cout << std::left << std::setw(20) << "operation" << std::right
            << std::setw(12) << "loop ms" << std::setw(12) << "simd ms"
            << std::setw(11) << "speedup" << std::setw(12) << "flop/byte\n";
  std::cout << std::string(67, '-') << "\n";

  // Keep x in [0.5, 1.0) so x^degree stays finite even at degree 32.
  Tensor x = make_f32({n}, 0.0f);
  {
    float *xp = static_cast<float *>(x.data());
    for (size_t i = 0; i < x.numel(); ++i) {
      xp[i] = 0.5f + static_cast<float>(i % 500) * 0.001f;
    }
  }
  for (size_t degree : {1u, 2u, 4u, 8u, 16u, 32u, 64u}) {
    run_fused_bench(x, degree, warmup, iters);
  }

  std::cout << "\nDone.\n";
  return 0;
}
