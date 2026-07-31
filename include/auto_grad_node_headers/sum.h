#pragma once

#include "auto_grad_node_headers/node.h"

#include <string>

namespace mango {

class SumBackward : public Node {
public:
  SumBackward(Tensor a);

  void backwardPass(const Tensor &grad_out) override;
  std::string function() const override;
};

} // namespace mango
