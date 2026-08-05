#include "auto_grad_node_headers/sq.h"

namespace mango {

SquareBackward::SquareBackward(Tensor a) { add_parent(std::move(a)); }

void SquareBackward::backwardPass(const Tensor &grad_out) {
  propagate(parents_[0], grad_out.mult_nr(parents_[0].add_nr(parents_[0])));
}

std::string SquareBackward::function() const { return "Square_fn"; }

} // namespace mango
