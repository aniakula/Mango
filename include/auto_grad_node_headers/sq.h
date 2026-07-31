#pragma once

#include "auto_grad_node_headers/node.h"

#include <string>

namespace mango {

class SquareBackward : public Node {
public:
  SquareBackward(Tensor a);

  void backwardPass(const Tensor &grad_out) override;
  std::string function() const override;
};

} // namespace mango