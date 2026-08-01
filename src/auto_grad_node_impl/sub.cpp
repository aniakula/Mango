#include "auto_grad_node_headers/sub.h"

#include "tensor_lib_headers/tensor.h"

namespace mango {

SubBackward::SubBackward(Tensor a, Tensor b) {
  parents_.push_back(std::move(a));
  parents_.push_back(std::move(b));
}

void SubBackward::backwardPass(const Tensor &grad_out) {
  parents_[0].accum_grad(grad_out);
  parents_[1].accum_grad(grad_out.negate_nr());

  if (parents_[0].grad_fn()) {
    parents_[0].grad_fn()->backwardPass(*parents_[0].grad());
  }
  if (parents_[1].grad_fn()) {
    parents_[1].grad_fn()->backwardPass(*parents_[1].grad());
  }
}

std::string SubBackward::function() const { return "Subtraction_fn"; }

} // namespace mango
