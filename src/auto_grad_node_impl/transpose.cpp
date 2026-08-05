#include "auto_grad_node_headers/transpose.h"

namespace mango {

TransposeBackward::TransposeBackward(Tensor a, size_t dim1, size_t dim2)
    : dim1_(dim1), dim2_(dim2) {
  add_parent(std::move(a));
}

void TransposeBackward::backwardPass(const Tensor &grad_out) {
  propagate(parents_[0],
            grad_out.transpose_nr(dim1_, dim2_).contiguous());
}

std::string TransposeBackward::function() const { return "transpose_fn"; }

} // namespace mango
