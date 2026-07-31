#pragma once

#include "tensor_lib_headers/tensor.h"

#include <string>
#include <vector>

namespace mango {

class Node {
public:
  virtual ~Node() = default;
  virtual void backwardPass(const Tensor &grad_out) = 0;
  virtual std::string function() const = 0;

  const std::vector<Tensor> &parents() const { return parents_; }

protected:
  std::vector<Tensor> parents_;
};

} // namespace mango
