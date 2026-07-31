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

  std::cout << "\nDone.\n";
  return 0;
}
