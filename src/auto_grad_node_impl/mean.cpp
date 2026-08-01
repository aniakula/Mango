#include "auto_grad_node_headers/mean.h"

namespace mango {

MeanBackward::MeanBackward(Tensor a) { parents_.push_back(std::move(a)); }

void MeanBackward::backwardPass(const Tensor &grad_out) {
  Tensor meanDerivative({1.0 / static_cast<double>(parents_[0].numel())},
                        Shape{});
  parents_[0].accum_grad(grad_out.mult_nr(meanDerivative));
  if (parents_[0].grad_fn()) {
    parents_[0].grad_fn()->backwardPass(*parents_[0].grad());
  }
}

std::string MeanBackward::function() const { return "Mean_fn"; }

} // namespace mango
