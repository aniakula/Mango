#include "tensor_test_utils.h"

#include "auto_grad_node_headers/softmax.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace mango {
namespace {

using test::expect_grad_near;

TEST(TensorAutograd, AddBackwardPropagatesUnitGradientToBothInputs) {
  Tensor a({1.0, 2.0, 3.0}, Shape{3}, true);
  Tensor b({4.0, 5.0, 6.0}, Shape{3}, true);
  Tensor loss = (a + b).sum();
  loss.backward();

  expect_grad_near<double>(a, {1, 1, 1});
  expect_grad_near<double>(b, {1, 1, 1});
}

TEST(TensorAutograd, SubtractAndNegateBackwardApplyCorrectSigns) {
  Tensor a({1.0, 2.0}, Shape{2}, true);
  Tensor b({3.0, 4.0}, Shape{2}, true);
  Tensor loss = (a - b).sum();
  loss.backward();
  expect_grad_near<double>(a, {1, 1});
  expect_grad_near<double>(b, {-1, -1});

  a.zero_grad();
  Tensor negative_loss = (-a).sum();
  negative_loss.backward();
  expect_grad_near<double>(a, {-1, -1});
}

TEST(TensorAutograd, MultiplyBackwardUsesOppositeOperand) {
  Tensor a({2.0, 3.0, 4.0}, Shape{3}, true);
  Tensor b({5.0, 6.0, 7.0}, Shape{3}, true);
  Tensor loss = (a * b).sum();
  loss.backward();

  expect_grad_near<double>(a, {5, 6, 7});
  expect_grad_near<double>(b, {2, 3, 4});
}

TEST(TensorAutograd, ScalarBroadcastBackwardReducesGradientToScalar) {
  Tensor vector({1.0, 2.0, 3.0, 4.0}, Shape{2, 2}, true);
  Tensor scalar({2.0}, Shape{}, true);
  Tensor loss = (vector * scalar + scalar).sum();
  loss.backward();

  expect_grad_near<double>(vector, {2, 2, 2, 2});
  // d(sum(vector * scalar + scalar))/dscalar = sum(vector) + 4
  expect_grad_near<double>(scalar, {14});
}

TEST(TensorAutograd, SquareBackwardComputesTwoX) {
  Tensor value({-2.0, 0.0, 3.0}, Shape{3}, true);
  Tensor loss = value.square().sum();
  loss.backward();
  expect_grad_near<double>(value, {-4, 0, 6});
}

TEST(TensorAutograd, ReluBackwardMasksNegativeAndZeroInputs) {
  Tensor value({-2.0, 0.0, 3.0, 4.0}, Shape{4}, true);
  Tensor loss = value.relu().sum();
  loss.backward();
  expect_grad_near<double>(value, {0, 0, 1, 1});
}

TEST(TensorAutograd, SumBackwardBroadcastsScalarGradientToEveryElement) {
  Tensor value({1.0, 2.0, 3.0, 4.0}, Shape{2, 2}, true);
  value.sum().backward();
  expect_grad_near<double>(value, {1, 1, 1, 1});
}

TEST(TensorAutograd, MeanBackwardWorksForFloatAndDouble) {
  Tensor doubles({1.0, 2.0, 3.0, 4.0}, Shape{4}, true);
  doubles.mean().backward();
  expect_grad_near<double>(doubles, {0.25, 0.25, 0.25, 0.25});

  Tensor floats({1.0f, 2.0f}, Shape{2}, true);
  floats.mean().backward();
  expect_grad_near<float>(floats, {0.5f, 0.5f}, 1e-6);
}

TEST(TensorAutograd, MatMulBackwardComputesBothMatrixGradients) {
  Tensor a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3}, true);
  Tensor b({7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, Shape{3, 2}, true);
  Tensor loss = a.matmul(b).sum();
  loss.backward();

  expect_grad_near<double>(a, {15, 19, 23, 15, 19, 23});
  expect_grad_near<double>(b, {5, 5, 7, 7, 9, 9});
}

TEST(TensorAutograd, TransposeBackwardUsesOriginalDimensions) {
  Tensor value({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, Shape{2, 2, 2}, true);
  Tensor weights({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, Shape{2, 2, 2});

  Tensor loss = (value.transpose(0, 2) * weights).sum();
  loss.backward();
  expect_grad_near<double>(value, {1, 5, 3, 7, 2, 6, 4, 8});
}

TEST(TensorAutograd, ChainedOperationsApplyChainRule) {
  Tensor x({-2.0, 1.0, 3.0}, Shape{3}, true);
  Tensor scale({2.0}, Shape{});
  Tensor loss = ((x * scale).square().relu()).mean();
  loss.backward();

  // mean((2x)^2): derivative is 8x / 3
  expect_grad_near<double>(x, {-16.0 / 3.0, 8.0 / 3.0, 8.0}, 1e-10);
}

TEST(TensorAutograd, RepeatedBackwardAccumulatesAndZeroGradResetsLeaf) {
  Tensor x({2.0, 3.0}, Shape{2}, true);
  Tensor loss = x.square().sum();
  loss.backward();
  expect_grad_near<double>(x, {4, 6});
  x.zero_grad();
  expect_grad_near<double>(x, {0, 0});
}

TEST(TensorAutograd, NonRecordingOpsDoNotCreateGraph) {
  Tensor x({1.0, 2.0}, Shape{2}, true);
  Tensor result = x.add_nr(x).square_nr().sum_nr();
  EXPECT_FALSE(result.requires_grad());
  EXPECT_EQ(result.grad_fn(), nullptr);
  EXPECT_THROW(result.backward(), std::logic_error);
}

TEST(TensorAutograd, SoftmaxBackwardMatchesJacobianVectorProduct) {
  Tensor input({0.0, 1.0, 2.0}, Shape{3}, true);
  Tensor output({0.09003057317038046, 0.24472847105479764, 0.6652409557748219},
                Shape{3});
  SoftmaxBackward backward(input, output);
  backward.backwardPass(Tensor({1.0, 2.0, 3.0}, Shape{3}));

  expect_grad_near<double>(
      input, {-0.141817093609, -0.140770357469, 0.282587451078}, 1e-10);
}

TEST(TensorAutograd, SharedLeafReceivesEveryPathContribution) {
  Tensor x({2.0, 3.0}, Shape{2}, true);
  Tensor loss = (x * x + x).sum();
  loss.backward();
  // d(x*x + x)/dx = 2x + 1
  expect_grad_near<double>(x, {5, 7});
}

TEST(TensorAutograd,
     SharedIntermediateRunsBackwardAfterAllConsumersContribute) {
  Tensor x({2.0, 3.0}, Shape{2}, true);
  Tensor squared = x.square();
  Tensor loss = (squared + squared).sum();
  loss.backward();
  // d(2 * x^2)/dx = 4x. The shared square node must execute only once.
  expect_grad_near<double>(x, {8, 12});
}

} // namespace
} // namespace mango
