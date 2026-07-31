#include "tensor.h"

#include <iostream>

using mango::DType;
using mango::Shape;
using mango::Tensor;

static void section(const char *title) {
  std::cout << "\n========== " << title << " ==========\n";
}

int main() {
  section("zeros");
  Tensor z = Tensor::zeros({2, 3}, DType::F32);
  z.log();

  section("fill and view");
  {
    auto *p = static_cast<float *>(z.data());
    for (int i = 0; i < 6; ++i) {
      p[i] = static_cast<float>(i + 1);
    }
    std::cout << "after filling 1..6:\n";
    z.log();

    Tensor v = z.view({3, 2});
    std::cout << "view as (3, 2) — shares storage with z:\n";
    v.log();

    p[0] = 99.0f;
    std::cout << "after z[0] = 99, view should show 99:\n";
    v.log();
  }

  section("clone");
  Tensor c = z.clone();
  auto *cp = static_cast<float *>(c.data());
  cp[0] = -1.0f;
  std::cout << "clone after changing clone[0] = -1 (z unchanged):\n";
  std::cout << "z:\n";
  z.log();
  std::cout << "c:\n";
  c.log();

  section("clone with new shape");
  Tensor r = z.clone({6});
  std::cout << "z.clone({6}):\n";
  r.log();

  section("ones and transpose");
  Tensor m = Tensor::ones<float>({2, 4});
  auto *mp = static_cast<float *>(m.data());
  for (size_t i = 0; i < m.numel(); ++i) {
    mp[i] = static_cast<float>(i);
  }
  std::cout << "matrix (2, 4) with 0..7:\n";
  m.log();

  m.transpose(0, 1);
  std::cout << "after transpose(0, 1) — non-contiguous layout:\n";
  m.log();

  section("reshape");
  try {
    Tensor bad = z.view({2, 2});
    bad.log();
  } catch (const std::exception &e) {
    std::cout << "expected error for view(2,2) on numel 6: " << e.what()
              << "\n";
  }

  section("matmul");
  {
    Tensor A({1.f, 2.f, 3.f, 4.f}, Shape{2, 2});
    Tensor B({2.f, 0.f, 0.f, 2.f}, Shape{2, 2});
    Tensor C = A.matmul(B);
    std::cout << "A @ 2I =\n";
    C.log();

    Tensor Batched({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}, Shape{2, 2, 2});
    Tensor R = Batched.matmul(B);
    std::cout << "(2,2,2) @ (2,2) =\n";
    R.log();
  }

  section("scalar reductions");
  {
    Tensor values({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f,
                   12.f, 13.f, 14.f, 15.f, 16.f},
                  Shape{2, 2, 2, 2});
    std::cout << "sum (rank 0):\n";
    values.sum().log();
    std::cout << "mean (rank 0):\n";
    values.mean().log();
    std::cout << "max (rank 0):\n";
    values.max().log();
    std::cout << "min (rank 0):\n";
    values.min().log();
  }

  std::cout << "\nDone.\n";
  return 0;
}
