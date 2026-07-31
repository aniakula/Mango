#include "tensor_lib_headers/shape.h"

namespace mango {

Shape::Shape(std::initializer_list<size_t> dims) : dims_(dims) {}
Shape::Shape(std::vector<size_t> dims) : dims_(dims) {}

size_t Shape::rank() const { return dims_.size(); }

size_t Shape::numel() const {
  size_t elems = 1;
  for (int dim : dims_) {
    elems *= dim;
  }
  return elems;
}

size_t Shape::operator[](size_t i) const { return dims_[i]; }

std::vector<size_t> Shape::to_vector() const { return dims_; }

Shape Shape::compute_strides(const Shape &dims) {
  std::vector<size_t> strides(dims.rank());
  size_t step = 1;
  for (size_t i = dims.rank(); i > 0; --i) {
    const size_t dim = i - 1;
    strides[dim] = step;
    step *= dims[dim];
  }
  return Shape(strides);
}

} // namespace mango
