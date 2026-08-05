#include "auto_grad_node_headers/matmul.h"

#include <stdexcept>

namespace mango {

MatMulBackward::MatMulBackward(Tensor a, Tensor b) {
  add_parent(std::move(a));
  add_parent(std::move(b));
}

void MatMulBackward::backwardPass(const Tensor &grad_out) {
  // C = A @ B  =>  dA = dC @ B^T ,  dB = A^T @ dC
  Tensor b_T = parents_[1].clone();
  const size_t b_rank = b_T.shape().rank();
  if (b_rank < 2) {
    throw std::invalid_argument("MatMulBackward: B must be at least 2D");
  }
  b_T.transpose_inplace(b_rank - 2, b_rank - 1);
  Tensor grad_a = grad_out.matmul_nr(b_T.contiguous());

  Tensor a_T = parents_[0].clone();
  const size_t a_rank = a_T.shape().rank();
  if (a_rank < 2) {
    throw std::invalid_argument("MatMulBackward: A must be at least 2D");
  }
  a_T.transpose_inplace(a_rank - 2, a_rank - 1);
  Tensor grad_b = a_T.contiguous().matmul_nr(grad_out);

  propagate(parents_[0], grad_a);
  propagate(parents_[1], grad_b);
}

std::string MatMulBackward::function() const {
  return "Matrix_Multiplication_fn";
}

} // namespace mango
