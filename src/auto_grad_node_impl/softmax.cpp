#include "auto_grad_node_headers/softmax.h"

#include <stdexcept>

namespace mango {

SoftmaxBackward::SoftmaxBackward(Tensor softmax_out) {
  parents_.push_back(std::move(softmax_out));
}

void SoftmaxBackward::backwardPass(const Tensor &grad_out) {
  (void)grad_out;
  throw std::runtime_error("TODO: SoftmaxBackward::backwardPass");
}

std::string SoftmaxBackward::function() const { return "Softmax_fn"; }

} // namespace mango
