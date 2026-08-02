#include "auto_grad_node_headers/transpose.h"

namespace mango {

TransposeBackward::TransposeBackward(Tensor a) {
  parents_.push_back(std::move(a));
}

void TransposeBackward::backwardPass(const Tensor &grad_out) {
  parents_[0].accum_grad(grad_out.transpose_nr());
  if (parents_[0].grad_fn()) {
    parents_[0].grad_fn()->backwardPass(*parents_[0].grad());
  }
}

std::string TransposeBackward::function() const { return "transpose_fn"; }

} // namespace mango
