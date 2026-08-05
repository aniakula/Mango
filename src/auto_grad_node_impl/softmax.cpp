#include "auto_grad_node_headers/softmax.h"

namespace mango {

SoftmaxBackward::SoftmaxBackward(Tensor input, Tensor softmax_out)
    : softmax_out_(std::move(softmax_out)) {
  add_parent(std::move(input));
}

void SoftmaxBackward::backwardPass(const Tensor &grad_out) {
  const Tensor &y = softmax_out_;
  propagate(parents_[0],
            y.mult_nr(grad_out.sub_nr((grad_out.mult_nr(y)).sum_nr())));
}

std::string SoftmaxBackward::function() const { return "Softmax_fn"; }

} // namespace mango
