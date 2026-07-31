#include "tensor_lib_headers/tensor.h"
#include "auto_grad_node_headers/add.h"
#include "auto_grad_node_headers/matmul.h"
#include "auto_grad_node_headers/mean.h"
#include "auto_grad_node_headers/mult.h"
#include "auto_grad_node_headers/node.h"
#include "auto_grad_node_headers/sq.h"
#include "auto_grad_node_headers/sub.h"
#include "auto_grad_node_headers/sum.h"
#include "tensor_lib_headers/tensor_internal.h"
#include "tensor_lib_headers/types.h"

#include <cstring>
#include <iomanip>
#include <stdexcept>

namespace mango {

Tensor::Tensor(Shape shape, DType type, bool learnable)
    : shape_(std::move(shape)), strides_(Shape::compute_strides(shape_)),
      offset_(0), dtype_(type), learnable_(learnable),
      storage_(std::make_shared<Storage>(shape_.numel() * dtype_size(type))),
      grad_tensor_(nullptr), grad_fn_(nullptr) {
  if (learnable_) {
    grad_tensor_ = std::make_shared<Tensor>(Tensor::zeros(shape_, dtype_));
  }
}

Tensor::~Tensor() = default;

void Tensor::clear_autograd() {
  learnable_ = false;
  grad_fn_.reset();
  grad_tensor_.reset();
}

Node *Tensor::grad_fn() { return grad_fn_.get(); }
const Tensor *Tensor::grad() { return grad_tensor_.get(); }
bool Tensor::requires_grad() const {
  return learnable_ || static_cast<bool>(grad_fn_);
}

Shape Tensor::shape() const { return shape_; }

Shape Tensor::strides() const { return strides_; }

DType Tensor::dtype() const { return dtype_; }

size_t Tensor::numel() const { return shape_.numel(); }

size_t Tensor::storage_offset() const { return offset_; }

Tensor Tensor::mean() const {
  if (numel() == 0) {
    throw std::invalid_argument("mean: empty tensor");
  }
  Tensor result = sum();
  detail::divide_scalar_inplace(result.data(), result.dtype(), numel());
  if (requires_grad()) {
    result.grad_fn_ = std::make_shared<MeanBackward>(*this);
  }
  return result;
}

Tensor Tensor::sum() const {
  Tensor input = contiguous();
  Tensor result(Shape{}, dtype_, /*learnable=*/false);
  detail::dispatch_reduction(input.data(), result.data(), input.numel(), dtype_,
                             detail::Reduction::Sum);
  if (requires_grad()) {
    result.grad_fn_ = std::make_shared<SumBackward>(*this);
  }
  return result;
}

Tensor Tensor::max() const {
  Tensor input = contiguous();
  Tensor result(Shape{}, dtype_, /*learnable=*/false);
  detail::dispatch_reduction(input.data(), result.data(), input.numel(), dtype_,
                             detail::Reduction::Max);
  return result;
}

Tensor Tensor::min() const {
  Tensor input = contiguous();
  Tensor result(Shape{}, dtype_, /*learnable=*/false);
  detail::dispatch_reduction(input.data(), result.data(), input.numel(), dtype_,
                             detail::Reduction::Min);
  return result;
}

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
  out.clear_autograd();
  return out;
}

Tensor Tensor::clone(const Shape &newShape) const {
  Tensor out = clone();
  out.reshape(newShape);
  return out;
}

bool Tensor::is_contiguous() const {
  return detail::is_contiguous_layout(shape_, strides_);
}

Tensor Tensor::contiguous() const {
  const size_t elem_size = dtype_size(dtype_);
  const size_t n = numel();
  const size_t nbytes = n * elem_size;

  if (is_contiguous()) {
    if (offset_ == 0 && storage_->bytes() == nbytes) {
      return *this;
    }
    Tensor out(shape_, dtype_);
    if (nbytes > 0) {
      std::memcpy(out.data(), data(), nbytes);
    }
    return out;
  }

  Tensor out(shape_, dtype_);
  if (nbytes > 0) {
    detail::copy_strided_to_contiguous(out.data(), data(), n, elem_size, shape_,
                                       strides_);
  }
  return out;
}

void Tensor::accum_grad(const Tensor &grad_out) {
  if (!grad_tensor_) {
    grad_tensor_ = std::make_shared<Tensor>(Tensor::zeros(shape_, dtype_));
  }
  *grad_tensor_ += grad_out;
}

void Tensor::zero_grad() {
  if (!grad_tensor_) {
    return;
  }
  *grad_tensor_ = Tensor::zeros(shape_, dtype_);
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

//---- Non-recording helpers ----

Tensor Tensor::add(const Tensor &other) const {
  Tensor out = detail::elementwise_binary(*this, other,
                                          [](auto a, auto b) { return a + b; });
  out.clear_autograd();
  return out;
}

Tensor Tensor::sub(const Tensor &other) const {
  Tensor out = detail::elementwise_binary(*this, other,
                                          [](auto a, auto b) { return a - b; });
  out.clear_autograd();
  return out;
}

Tensor Tensor::mult(const Tensor &other) const {
  Tensor out = detail::elementwise_binary(*this, other,
                                          [](auto a, auto b) { return a * b; });
  out.clear_autograd();
  return out;
}

Tensor Tensor::neg() const {
  if (dtype_ == DType::B) {
    throw std::invalid_argument("neg: bool tensors are not supported");
  }

  Tensor src = contiguous();
  Tensor out(src.shape(), src.dtype(), /*learnable=*/false);
  const size_t n = src.numel();

  switch (dtype_) {
  case DType::F32: {
    const auto *in = static_cast<const float *>(src.data());
    auto *dest = static_cast<float *>(out.data());
    for (size_t i = 0; i < n; ++i) {
      dest[i] = -in[i];
    }
    break;
  }
  case DType::F64: {
    const auto *in = static_cast<const double *>(src.data());
    auto *dest = static_cast<double *>(out.data());
    for (size_t i = 0; i < n; ++i) {
      dest[i] = -in[i];
    }
    break;
  }
  case DType::I32: {
    const auto *in = static_cast<const int32_t *>(src.data());
    auto *dest = static_cast<int32_t *>(out.data());
    for (size_t i = 0; i < n; ++i) {
      dest[i] = -in[i];
    }
    break;
  }
  case DType::I64: {
    const auto *in = static_cast<const int64_t *>(src.data());
    auto *dest = static_cast<int64_t *>(out.data());
    for (size_t i = 0; i < n; ++i) {
      dest[i] = -in[i];
    }
    break;
  }
  case DType::B:
    break;
  }
  return out;
}

Tensor Tensor::mm(const Tensor &other) const {
  Tensor out = detail::tensorMatMul(*this, other);
  out.clear_autograd();
  return out;
}

Tensor Tensor::sq() const { return mult(*this); }

//---- Recording ops ----

Tensor Tensor::operator+(const Tensor &other) const {
  Tensor out = add(other);
  if (requires_grad() || other.requires_grad()) {
    out.grad_fn_ = std::make_shared<AddBackward>(*this, other);
  }
  return out;
}

Tensor Tensor::operator-() const {
  Tensor out = neg();
  if (requires_grad()) {
    Tensor zero = Tensor::zeros(shape_, dtype_);
    out.grad_fn_ = std::make_shared<SubBackward>(std::move(zero), *this);
  }
  return out;
}

Tensor Tensor::operator-(const Tensor &other) const {
  Tensor out = sub(other);
  if (requires_grad() || other.requires_grad()) {
    out.grad_fn_ = std::make_shared<SubBackward>(*this, other);
  }
  return out;
}

Tensor Tensor::operator*(const Tensor &other) const {
  Tensor out = mult(other);
  if (requires_grad() || other.requires_grad()) {
    out.grad_fn_ = std::make_shared<MulBackward>(*this, other);
  }
  return out;
}

Tensor &Tensor::operator+=(const Tensor &other) {
  detail::elementwise_inplace(*this, other,
                              [](auto a, auto b) { return a + b; });
  return *this;
}

Tensor &Tensor::operator-=(const Tensor &other) {
  detail::elementwise_inplace(*this, other,
                              [](auto a, auto b) { return a - b; });
  return *this;
}

Tensor &Tensor::operator*=(const Tensor &other) {
  detail::elementwise_inplace(*this, other,
                              [](auto a, auto b) { return a * b; });
  return *this;
}

Tensor Tensor::matmul(const Tensor &other) const {
  Tensor out = mm(other);
  if (requires_grad() || other.requires_grad()) {
    out.grad_fn_ = std::make_shared<MatMulBackward>(*this, other);
  }
  return out;
}

Tensor Tensor::square() const {
  Tensor out = sq();
  if (requires_grad()) {
    out.grad_fn_ = std::make_shared<SquareBackward>(*this);
  }
  return out;
}

void Tensor::log(std::ostream &os) const {
  os << "Tensor dtype=" << detail::dtype_name(dtype_) << " shape=";
  detail::print_shape(os, shape_);
  os << " strides=";
  detail::print_shape(os, strides_);
  os << " offset=" << offset_;
  os << " contiguous=" << (is_contiguous() ? "true" : "false");
  os << "\n";

  if (numel() == 0) {
    os << "  (empty)\n";
    return;
  }

  const size_t elem_size = dtype_size(dtype_);
  const auto *base =
      static_cast<const char *>(storage_->data()) + offset_ * elem_size;
  const size_t row_width = shape_.rank() == 0 ? 1 : shape_[shape_.rank() - 1];

  const auto print_row = [&](auto tag) {
    using T = decltype(tag);
    for (size_t flat = 0; flat < numel(); ++flat) {
      size_t storage_idx = flat;
      if (!is_contiguous()) {
        storage_idx = detail::linear_index(flat, shape_, strides_);
      }
      const T value =
          *reinterpret_cast<const T *>(base + storage_idx * elem_size);
      if (flat > 0 && flat % row_width == 0) {
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
      if (!is_contiguous()) {
        storage_idx = detail::linear_index(flat, shape_, strides_);
      }
      const bool value = base[storage_idx] != 0;
      if (flat > 0 && flat % row_width == 0) {
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
