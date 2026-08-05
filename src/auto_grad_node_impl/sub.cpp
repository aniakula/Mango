#include "auto_grad_node_headers/sub.h"

#include "tensor_lib_headers/tensor.h"

namespace mango {

SubBackward::SubBackward(Tensor a, Tensor b) {
  add_parent(std::move(a));
  add_parent(std::move(b));
}

void SubBackward::backwardPass(const Tensor &grad_out) {
  propagate(parents_[0], grad_out);
  propagate(parents_[1], grad_out.negate_nr());
}

std::string SubBackward::function() const { return "Subtraction_fn"; }

} // namespace mango
