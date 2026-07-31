#pragma once

#include "auto_grad_node_headers/node.h"

#include <string>

namespace mango {

class SoftmaxBackward : public Node {
public:
  // Saves the softmax output used in the Jacobian-vector product.
  explicit SoftmaxBackward(Tensor softmax_out);

  void backwardPass(const Tensor &grad_out) override;
  std::string function() const override;
};

} // namespace mango
