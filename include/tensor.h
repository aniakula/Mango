#pragma once

#include "shape.h"
#include "storage.h"
#include "types.h"

#include <cstddef>
#include <iostream>
#include <memory>

namespace mango {

class Tensor {
public:
  Tensor(Shape shape, DType type);

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

  static Tensor zeros(const Shape &shape, DType type = DType::F32);

  template <typename T> static Tensor ones(const Shape &shape);

  void transpose(size_t dim1 = 0, size_t dim2 = 1);

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

} // namespace mango
