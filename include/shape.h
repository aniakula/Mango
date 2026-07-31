#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace mango {

class Shape {
public:
  Shape(std::initializer_list<size_t> dims);
  Shape(std::vector<size_t> dims);

  size_t rank() const;
  size_t numel() const;
  size_t operator[](size_t i) const;
  std::vector<size_t> to_vector() const;

  static Shape compute_strides(const Shape &dims);

private:
  std::vector<size_t> dims_;
};

} // namespace mango
