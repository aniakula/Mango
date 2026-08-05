#include "auto_grad_node_headers/mult.h"

namespace mango {

MulBackward::MulBackward(Tensor a, Tensor b) {
  add_parent(std::move(a));
  add_parent(std::move(b));
}

void MulBackward::backwardPass(const Tensor &grad_out) {
  propagate(parents_[0], grad_out.mult_nr(parents_[1]));
  propagate(parents_[1], grad_out.mult_nr(parents_[0]));
}

std::string MulBackward::function() const { return "Multiplication_fn"; }

} // namespace mango
