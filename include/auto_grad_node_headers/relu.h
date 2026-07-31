#pragma once

#include "auto_grad_node_headers/node.h"

#include <string>

namespace mango {

class ReluBackward : public Node {
public:
  // Saves the forward input (or mask) for the local gradient.
  explicit ReluBackward(Tensor input);

  void backwardPass(const Tensor &grad_out) override;
  std::string function() const override;
};

} // namespace mango
