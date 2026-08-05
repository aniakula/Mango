#include "auto_grad_node_headers/add.h"
namespace mango {

AddBackward::AddBackward(Tensor a, Tensor b) {
  add_parent(std::move(a));
  add_parent(std::move(b));
}

void AddBackward::backwardPass(const Tensor &grad_out) {
  propagate(parents_[0], grad_out);
  propagate(parents_[1], grad_out);
}

std::string AddBackward::function() const { return "Addition_fn"; }

} // namespace mango
