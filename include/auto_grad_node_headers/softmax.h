#pragma once

#include "auto_grad_node_headers/node.h"

#include <string>

namespace mango {

class SoftmaxBackward : public Node {
public:
  // input receives the gradient; softmax_out (y) is saved for the Jacobian.
  SoftmaxBackward(Tensor input, Tensor softmax_out);

  void backwardPass(const Tensor &grad_out) override;
  std::string function() const override;

private:
  Tensor softmax_out_;
};

} // namespace mango
