#include "auto_grad_node_headers/softmax.h"

namespace mango {

SoftmaxBackward::SoftmaxBackward(Tensor input, Tensor softmax_out)
    : softmax_out_(std::move(softmax_out)) {
  parents_.push_back(std::move(input));
}

void SoftmaxBackward::backwardPass(const Tensor &grad_out) {
  // dx = y * (g - sum(g * y))  where y = softmax(x), g = grad_out
  const Tensor &y = softmax_out_;
  parents_[0].accum_grad(
      y.mult_nr(grad_out.sub_nr((grad_out.mult_nr(y)).sum())));
  if (parents_[0].grad_fn()) {
    parents_[0].grad_fn()->backwardPass(*parents_[0].grad());
  }
}

std::string SoftmaxBackward::function() const { return "Softmax_fn"; }

} // namespace mango
