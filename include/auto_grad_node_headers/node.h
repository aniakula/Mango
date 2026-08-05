#pragma once

#include "tensor_lib_headers/tensor.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace mango {

class Node {
public:
  virtual ~Node() = default;
  virtual void backwardPass(const Tensor &grad_out) = 0;
  virtual std::string function() const = 0;
  const std::vector<Tensor> &parents() const { return parents_; }

protected:
  void add_parent(Tensor parent) {
    if (parent.grad_fn()) {
      parent.grad_fn()->consumer_count_++;
    }
    parents_.push_back(std::move(parent));
  }

  static void propagate(Tensor &parent, const Tensor &gradient) {
    parent.accum_grad(gradient);
    Node *grad_fn = parent.grad_fn();
    if (!grad_fn) {
      return;
    }
    grad_fn->received_gradients_++;
    if (grad_fn->received_gradients_ == grad_fn->consumer_count_) {
      grad_fn->backwardPass(*parent.grad());
    }
  }

  std::vector<Tensor> parents_;

private:
  size_t consumer_count_ = 0;
  size_t received_gradients_ = 0;
};

} // namespace mango
