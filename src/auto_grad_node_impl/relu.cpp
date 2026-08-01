#include "auto_grad_node_headers/relu.h"

#include <stdexcept>

namespace mango {

ReluBackward::ReluBackward(Tensor input) {
  parents_.push_back(std::move(input));
}

void ReluBackward::backwardPass(const Tensor &grad_out) {
  const Tensor &x = parents_[0];
  Tensor local = grad_out.clone(); 
  const size_t n = x.numel();
  switch (x.dtype()) {
  case DType::F32: {
    const float *xp = static_cast<const float *>(x.data());
    float *gp = static_cast<float *>(local.data());
    for (size_t i = 0; i < n; ++i) {
      if (!(xp[i] > 0.f))
        gp[i] = 0.f;
    }
    break;
  }
  case DType::F64: {
    const double *xp = static_cast<const double *>(x.data());
    double *gp = static_cast<double *>(local.data());
    for (size_t i = 0; i < n; ++i) {
      if (!(xp[i] > 0.0))
        gp[i] = 0.0;
    }
    break;
  }
  default:
    throw std::invalid_argument("relu backward: unsupported dtype");
  }
  parents_[0].accum_grad(local);
  if (parents_[0].grad_fn()) {
    parents_[0].grad_fn()->backwardPass(*parents_[0].grad());
  }
}

std::string ReluBackward::function() const { return "Relu_fn"; }

} // namespace mango
