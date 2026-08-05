#include "auto_grad_node_headers/sum.h"

namespace mango {

SumBackward::SumBackward(Tensor a) { add_parent(std::move(a)); }

void SumBackward::backwardPass(const Tensor &grad_out) {
  propagate(parents_[0], grad_out);
}

std::string SumBackward::function() const { return "Sum_fn"; }

} // namespace mango
