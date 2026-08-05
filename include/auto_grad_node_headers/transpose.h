#pragma once

#include "auto_grad_node_headers/node.h"

#include <string>

namespace mango {

class TransposeBackward : public Node {
public:
  TransposeBackward(Tensor input, size_t dim1, size_t dim2);

  void backwardPass(const Tensor &grad_out) override;
  std::string function() const override;

private:
  size_t dim1_;
  size_t dim2_;
};

} // namespace mango
