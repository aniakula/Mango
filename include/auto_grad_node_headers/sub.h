#pragma once

#include "auto_grad_node_headers/node.h"

#include <string>

namespace mango {

class SubBackward : public Node {
public:
  SubBackward(Tensor a, Tensor b);

  void backwardPass(const Tensor &grad_out) override;
  std::string function() const override;
};

} // namespace mango
