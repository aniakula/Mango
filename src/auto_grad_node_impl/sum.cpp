#include "auto_grad_node_headers/sum.h"

namespace mango {

SumBackward::SumBackward(Tensor a) { parents_.push_back(std::move(a)); }

void SumBackward::backwardPass(const Tensor &grad_out) {
  parents_[0].accum_grad(grad_out);
  if (parents_[0].grad_fn()) {
    parents_[0].grad_fn()->backwardPass(*parents_[0].grad());
  }
}

std::string SumBackward::function() const { return "Sum_fn"; }

} // namespace mango
