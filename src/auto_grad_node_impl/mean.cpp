#include "auto_grad_node_headers/mean.h"

namespace mango {

MeanBackward::MeanBackward(Tensor a) { add_parent(std::move(a)); }

void MeanBackward::backwardPass(const Tensor &grad_out) {
  Tensor meanDerivative =
      parents_[0].dtype() == DType::F32
          ? Tensor({1.0f / static_cast<float>(parents_[0].numel())}, Shape{})
          : Tensor({1.0 / static_cast<double>(parents_[0].numel())}, Shape{});
  propagate(parents_[0], grad_out.mult_nr(meanDerivative));
}

std::string MeanBackward::function() const { return "Mean_fn"; }

} // namespace mango
