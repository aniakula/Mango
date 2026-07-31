#pragma once

#include "shape.h"
#include "storage.h"
#include "types.h"

#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace mango {

class Tensor {
public:
  Tensor(Shape shape, DType type);

  template <typename T>
  Tensor(std::initializer_list<T> data, const Shape &shape);

  Shape shape() const;
  Shape strides() const;
  DType dtype() const;
  size_t numel() const;
  size_t storage_offset() const;

  void *data();
  const void *data() const;

  void reshape(const Shape &newShape);
  Tensor view(const Shape &newShape) const;
  Tensor clone() const;
  Tensor clone(const Shape &newShape) const;
  bool is_contiguous() const;
  Tensor contiguous() const;

  static Tensor zeros(const Shape &shape, DType type = DType::F32);

  template <typename T> static Tensor ones(const Shape &shape);

  void transpose(size_t dim1 = 0, size_t dim2 = 1);

  Tensor operator+(const Tensor &other) const;
  Tensor operator-(const Tensor &other) const;
  Tensor operator*(const Tensor &other) const;
  Tensor &operator+=(const Tensor &other);
  Tensor &operator-=(const Tensor &other);
  Tensor &operator*=(const Tensor &other);

  Tensor matmul(const Tensor &other) const;
  inline Tensor matmul(const Tensor &a, const Tensor &b) const {
    return a.matmul(b);
  }

  void log(std::ostream &os = std::cout) const;

private:
  Shape shape_;
  Shape strides_;
  size_t offset_;
  DType dtype_;
  std::shared_ptr<Storage> storage_;
};

template <typename T> Tensor Tensor::ones(const Shape &shape) {
  Tensor tensor(shape, type_of<T>());
  T *p = static_cast<T *>(tensor.data());
  const T one = static_cast<T>(1);
  for (size_t i = 0; i < tensor.numel(); ++i) {
    p[i] = one;
  }
  return tensor;
}

template <typename T>
Tensor::Tensor(std::initializer_list<T> data, const Shape &shape)
    : Tensor(shape, type_of<T>()) {
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
