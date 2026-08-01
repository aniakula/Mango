#include "auto_grad_node_headers/sq.h"

namespace mango {

SquareBackward::SquareBackward(Tensor a) { parents_.push_back(std::move(a)); }

void SquareBackward::backwardPass(const Tensor &grad_out) {
  parents_[0].accum_grad(grad_out.mult_nr(parents_[0].add_nr(parents_[0])));
  if (parents_[0].grad_fn()) {
    parents_[0].grad_fn()->backwardPass(*parents_[0].grad());
  }
}

std::string SquareBackward::function() const { return "Square_fn"; }

} // namespace mango
