#include "tensor_lib_headers/shape.h"
#include "tensor_lib_headers/tensor.h"
#include "tensor_lib_headers/types.h"
#include <cstddef>
#include <iostream>

using mango::DType;
using mango::Shape;
using mango::Tensor;

static void section(const char *title) {
  std::cout << "\n========== " << title << " ==========\n";
}

Tensor forward(const Tensor &X, const Tensor &W, const Tensor &b) {
  return W * (X) + b;
}

int main() {
  // section("zeros");
  // Tensor z = Tensor::zeros({2, 3}, DType::F32);
  // z.log();

  // section("fill and view");
  // {
  //   auto *p = static_cast<float *>(z.data());
  //   for (int i = 0; i < 6; ++i) {
  //     p[i] = static_cast<float>(i + 1);
  //   }
  //   std::cout << "after filling 1..6:\n";
  //   z.log();

  //   Tensor v = z.view({3, 2});
  //   std::cout << "view as (3, 2) — shares storage with z:\n";
  //   v.log();

  //   p[0] = 99.0f;
  //   std::cout << "after z[0] = 99, view should show 99:\n";
  //   v.log();
  // }

  // section("clone");
  // Tensor c = z.clone();
  // auto *cp = static_cast<float *>(c.data());
  // cp[0] = -1.0f;
  // std::cout << "clone after changing clone[0] = -1 (z unchanged):\n";
  // std::cout << "z:\n";
  // z.log();
  // std::cout << "c:\n";
  // c.log();

  // section("clone with new shape");
  // Tensor r = z.clone({6});
  // std::cout << "z.clone({6}):\n";
  // r.log();

  // section("ones and transpose");
  // Tensor m = Tensor::ones<float>({2, 4});
  // auto *mp = static_cast<float *>(m.data());
  // for (size_t i = 0; i < m.numel(); ++i) {
  //   mp[i] = static_cast<float>(i);
  // }
  // std::cout << "matrix (2, 4) with 0..7:\n";
  // m.log();

  // m.transpose(0, 1);
  // std::cout << "after transpose(0, 1) — non-contiguous layout:\n";
  // m.log();

  // section("reshape");
  // try {
  //   Tensor bad = z.view({2, 2});
  //   bad.log();
  // } catch (const std::exception &e) {
  //   std::cout << "expected error for view(2,2) on numel 6: " << e.what()
  //             << "\n";
  // }

  // section("matmul");
  // {
  //   Tensor A({1.f, 2.f, 3.f, 4.f}, Shape{2, 2});
  //   Tensor B({2.f, 0.f, 0.f, 2.f}, Shape{2, 2});
  //   Tensor C = A.matmul(B);
  //   std::cout << "A @ 2I =\n";
  //   C.log();

  //   Tensor Batched({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}, Shape{2, 2, 2});
  //   Tensor R = Batched.matmul(B);
  //   std::cout << "(2,2,2) @ (2,2) =\n";
  //   R.log();
  // }

  // section("scalar reductions");
  // {
  //   Tensor values({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f,
  //                  12.f, 13.f, 14.f, 15.f, 16.f},
  //                 Shape{2, 2, 2, 2});
  //   std::cout << "sum (rank 0):\n";
  //   values.sum().log();
  //   std::cout << "mean (rank 0):\n";
  //   values.mean().log();
  //   std::cout << "max (rank 0):\n";
  //   values.max().log();
  //   std::cout << "min (rank 0):\n";
  //   values.min().log();
  // }

  // section("scalar broadcast");
  // {
  //   Tensor x({1.f, 2.f, 3.f, 4.f}, Shape{2, 2});
  //   Tensor two({2.f}, Shape{});
  //   std::cout << "x + 2 (scalar tensor):\n";
  //   (x + two).log();
  //   std::cout << "2 * x:\n";
  //   (two * x).log();
  //   std::cout << "x -= mean(x):\n";
  //   x -= x.mean();
  //   x.log();
  // }

  // std::cout << "\nDone.\n";

  Tensor y = Tensor({2.0, 4.0, 6.0, 8.0, 10.0}, Shape{5});
  Tensor x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0}, Shape{5});
  Tensor W = Tensor::randn<double>(Shape{1}, true);
  Tensor b = Tensor::randn<double>(Shape{}, true);
  Tensor y_hat = forward(x, W, b);
  y_hat.log();
  Tensor loss = ((y - y_hat).square()).mean();
  for (size_t epoch = 0; epoch < 30; epoch++) {
    y_hat = forward(x, W, b);
    loss = ((y - y_hat).square()).mean();
    std::cout << "loss: " << *static_cast<double *>(loss.data())
              << " epoch: " << epoch + 1 << std::endl;

    loss.backward();
    W -= W.grad()->mult_nr(Tensor({0.02}, Shape{}));
    b -= b.grad()->mult_nr(Tensor({0.02}, Shape{}));

    W.zero_grad();
    b.zero_grad();
  }

  y_hat = forward(x, W, b);
  y_hat.log();
  W.log();
  b.log();

  W = W.to(DType::I32);
  b = b.to(DType::I32);

  W.log();
  b.log();

  return 0;
}
