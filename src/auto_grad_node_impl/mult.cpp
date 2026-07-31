#include "auto_grad_node_headers/mult.h"

namespace mango {

MulBackward::MulBackward(Tensor a, Tensor b) {
  parents_.push_back(std::move(a));
  parents_.push_back(std::move(b));
}

void MulBackward::backwardPass(const Tensor &grad_out) {
  parents_[0].accum_grad(grad_out.mult(parents_[1]));
  parents_[1].accum_grad(grad_out.mult(parents_[0]));
  if (parents_[0].grad_fn()) {
    parents_[0].grad_fn()->backwardPass(*parents_[0].grad());
  }

  if (parents_[1].grad_fn()) {
    parents_[1].grad_fn()->backwardPass(*parents_[1].grad());
  }
}

std::string MulBackward::function() const { return "Multiplication_fn"; }

} // namespace mango
