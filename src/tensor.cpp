#include "tensor.h"

#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

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

bool is_contiguous(const Shape &shape, const Shape &strides) {
  return strides.to_vector() == Shape::compute_strides(shape).to_vector();
}

size_t linear_index(size_t flat, const Shape &shape, const Shape &strides) {
  size_t idx = 0;
  size_t rem = flat;
  for (size_t d = 0; d < shape.rank(); ++d) {
    const size_t dim = shape[d];
    const size_t coord = dim == 0 ? 0 : rem % dim;
    rem = dim == 0 ? 0 : rem / dim;
    idx += coord * strides[d];
  }
  return idx;
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

void Tensor::log(std::ostream &os) const {
  os << "Tensor dtype=" << dtype_name(dtype_) << " shape=";
  print_shape(os, shape_);
  os << " strides=";
  print_shape(os, strides_);
  os << " offset=" << offset_;
  os << " contiguous=" << (is_contiguous(shape_, strides_) ? "true" : "false");
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
      if (!is_contiguous(shape_, strides_)) {
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
      if (!is_contiguous(shape_, strides_)) {
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
