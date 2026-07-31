#include "tensor_lib_headers/storage.h"

#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace mango {

Storage::Storage(size_t bytes) : bytes_(bytes), data_(nullptr) {
  if (bytes_ == 0) {
    return;
  }
  data_ = std::malloc(bytes_);
  if (data_ == nullptr) {
    throw std::bad_alloc();
  }
}

Storage::~Storage() { release(); }

Storage::Storage(Storage &&other) noexcept
    : bytes_(other.bytes_), data_(other.data_) {
  other.bytes_ = 0;
  other.data_ = nullptr;
}

Storage &Storage::operator=(Storage &&other) noexcept {
  if (this != &other) {
    release();
    bytes_ = other.bytes_;
    data_ = other.data_;
    other.bytes_ = 0;
    other.data_ = nullptr;
  }
  return *this;
}

void *Storage::data() { return data_; }

const void *Storage::data() const { return data_; }

size_t Storage::bytes() const { return bytes_; }

void Storage::release() {
  if (data_ != nullptr) {
    std::free(data_);
    data_ = nullptr;
  }
  bytes_ = 0;
}

} // namespace mango
