#pragma once
#include "shape.h"
#include "storage.h"
#include "types.h"
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>

namespace mango {
class Node;
class Tensor {
public:
  Tensor(Shape shape, DType type, bool learnable = false);
  ~Tensor();

  template <typename T>
  Tensor(std::initializer_list<T> data, const Shape &shape,
         bool learnable = false);

  Shape shape() const;
  Shape strides() const;
  DType dtype() const;
  Tensor to(DType &&new_type);
  size_t numel() const;
  size_t storage_offset() const;
  Node *grad_fn();
  const Tensor *grad();
  bool requires_grad() const;
  void *data();
  const void *data() const;

  void reshape(const Shape &newShape);
  Tensor view(const Shape &newShape) const;
  Tensor clone() const;
  Tensor clone(const Shape &newShape) const;
  bool is_contiguous() const;
  Tensor contiguous() const;
  void accum_grad(const Tensor &grad_out);
  void zero_grad();
  void backward();

  static Tensor zeros(const Shape &shape, DType type = DType::F32);

  template <typename T>
  static Tensor ones(const Shape &shape, bool learnable = false);
  template <typename T>
  static Tensor empty(const Shape &shape, bool learnable = false);
  template <typename T>
  static Tensor randn(const Shape &shape, bool learnable = false);

  void transpose(size_t dim1 = 0, size_t dim2 = 1);

  // Non-recording kernels (no grad_fn_)
  Tensor add(const Tensor &other) const;
  Tensor sub(const Tensor &other) const;
  Tensor mult(const Tensor &other) const;
  Tensor neg() const;
  Tensor mm(const Tensor &other) const;
  Tensor sq() const;

  // Recording ops
  Tensor operator+(const Tensor &other) const;
  Tensor operator-() const;
  Tensor operator-(const Tensor &other) const;
  Tensor operator*(const Tensor &other) const;
  Tensor &operator+=(const Tensor &other);
  Tensor &operator-=(const Tensor &other);
  Tensor &operator*=(const Tensor &other);

  Tensor matmul(const Tensor &other) const;
  Tensor square() const;
  inline Tensor matmul(const Tensor &a, const Tensor &b) const {
    return a.matmul(b);
  }

  Tensor mean() const;
  Tensor sum() const;
  Tensor max() const;
  Tensor min() const;

  void log(std::ostream &os = std::cout) const;

private:
  void clear_autograd();

  Shape shape_;
  Shape strides_;
  size_t offset_;
  DType dtype_;
  bool learnable_;
  std::shared_ptr<Storage> storage_;
  std::shared_ptr<Tensor> grad_tensor_;
  std::shared_ptr<Node> grad_fn_;
};

//----Template defs:----

template <typename T> Tensor Tensor::ones(const Shape &shape, bool learnable) {
  Tensor tensor(shape, type_of<T>(), learnable);
  T *p = static_cast<T *>(tensor.data());
  const T one = static_cast<T>(1);
  for (size_t i = 0; i < tensor.numel(); ++i) {
    p[i] = one;
  }
  return tensor;
}

template <typename T> Tensor Tensor::empty(const Shape &shape, bool learnable) {
  return Tensor(shape, type_of<T>(), learnable);
}

template <typename T> Tensor Tensor::randn(const Shape &shape, bool learnable) {
  static_assert(std::is_same<T, float>() || std::is_same<T, double>(),
                "Type must be float32 or float64 for random initialization");
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<T> dist(T(0), T(1));
  Tensor t = Tensor::empty<T>(shape, learnable);
  T *data_ptr = static_cast<T *>(t.data());
  for (size_t i = 0; i < t.numel(); i++) {
    data_ptr[i] = dist(gen);
  }
  return t;
}

template <typename T>
Tensor::Tensor(std::initializer_list<T> data, const Shape &shape,
               bool learnable)
    : Tensor(shape, type_of<T>(), learnable) {
  if (shape.numel() != data.size()) {
    std::string error_msg = std::format(
        "Tensor shape size mismatch: expected {} elements from shape, "
        "but initializer list provided {} elements.",
        shape.numel(), data.size());
    throw std::invalid_argument(error_msg);
  }

  T *data_ptr = static_cast<T *>(this->data());
  std::copy(data.begin(), data.end(), data_ptr);
}

} // namespace mango
