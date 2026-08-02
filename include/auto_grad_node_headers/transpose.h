#pragma once

#include "auto_grad_node_headers/node.h"

#include <string>

namespace mango {

class TransposeBackward : public Node {
public:
  // Saves the forward input (or mask) for the local gradient.
  explicit TransposeBackward(Tensor input);

  void backwardPass(const Tensor &grad_out) override;
  std::string function() const override;
};

} // namespace mango
