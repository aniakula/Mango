#include "tensor_test_utils.h"

#include "auto_grad_node_headers/node.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace mango {
namespace {

using test::expect_shape;
using test::expect_values;
using test::expect_values_near;

TEST(TensorNonRecordingElementwise, AddSubtractAndMultiplySameShape) {
  Tensor a({1.0f, -2.0f, 3.0f, 4.0f}, Shape{2, 2}, true);
  Tensor b({5.0f, 6.0f, -1.0f, 2.0f}, Shape{2, 2});

  Tensor added = a.add_nr(b);
  Tensor subtracted = a.sub_nr(b);
  Tensor multiplied = a.mult_nr(b);

  expect_values<float>(added, {6, 4, 2, 6});
  expect_values<float>(subtracted, {-4, -8, 4, 2});
  expect_values<float>(multiplied, {5, -12, -3, 8});
  EXPECT_FALSE(added.requires_grad());
  EXPECT_EQ(added.grad_fn(), nullptr);
  EXPECT_FALSE(subtracted.requires_grad());
  EXPECT_FALSE(multiplied.requires_grad());
}

TEST(TensorNonRecordingElementwise, SupportsHigherDimensionalIntegerTensors) {
  Tensor a({1, 2, 3, 4, 5, 6, 7, 8}, Shape{2, 2, 2});
  Tensor b({8, 7, 6, 5, 4, 3, 2, 1}, Shape{2, 2, 2});
  expect_values<int32_t>(a.add_nr(b), {9, 9, 9, 9, 9, 9, 9, 9});
  expect_values<int32_t>(a.mult_nr(b), {8, 14, 18, 20, 20, 18, 14, 8});
}

TEST(TensorNonRecordingElementwise, PreservesLogicalOrderOfNonContiguousInput) {
  Tensor transposed =
      Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3}).transpose_nr();
  Tensor rhs({10.0, 20.0, 30.0, 40.0, 50.0, 60.0}, Shape{3, 2});
  Tensor result = transposed.add_nr(rhs);

  EXPECT_TRUE(result.is_contiguous());
  expect_values<double>(result, {11, 24, 32, 45, 53, 66});
}

TEST(TensorScalarBroadcast, WorksForScalarOnEitherSide) {
  Tensor vector({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
  Tensor scalar({2.0}, Shape{});

  expect_values<double>(vector.add_nr(scalar), {3, 4, 5, 6});
  expect_values<double>(scalar.add_nr(vector), {3, 4, 5, 6});
  expect_values<double>(vector.sub_nr(scalar), {-1, 0, 1, 2});
  expect_values<double>(scalar.sub_nr(vector), {1, 0, -1, -2});
  expect_values<double>(vector.mult_nr(scalar), {2, 4, 6, 8});
  expect_values<double>(scalar.mult_nr(vector), {2, 4, 6, 8});
}

TEST(TensorScalarBroadcast, TreatsAnySingleElementTensorAsScalarLike) {
  Tensor vector({1.0f, 2.0f, 3.0f}, Shape{3});
  Tensor one_by_one({4.0f}, Shape{1, 1});
  Tensor result = vector.add_nr(one_by_one);
  expect_shape(result, {3});
  expect_values<float>(result, {5, 6, 7});
}

TEST(TensorInPlaceElementwise, SupportsMatchingShapesAndScalarRightSide) {
  Tensor value({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
  value += Tensor({1.0}, Shape{});
  expect_values<double>(value, {2, 3, 4, 5});
  value *= Tensor({2.0, 3.0, 4.0, 5.0}, Shape{2, 2});
  expect_values<double>(value, {4, 9, 16, 25});
  value -= Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
  expect_values<double>(value, {3, 7, 13, 21});
}

TEST(TensorInPlaceElementwise, UpdatesNonContiguousTensorInLogicalOrder) {
  Tensor value =
      Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3}).transpose_nr();
  value += Tensor({10.0, 20.0, 30.0, 40.0, 50.0, 60.0}, Shape{3, 2});
  expect_values<double>(value, {11, 24, 32, 45, 53, 66});
  value *= Tensor({2.0}, Shape{});
  expect_values<double>(value, {22, 48, 64, 90, 106, 132});
}

TEST(TensorElementwiseErrors, RejectsInvalidShapesTypesAndBool) {
  Tensor matrix({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
  Tensor wrong_shape({1.0f, 2.0f, 3.0f, 4.0f}, Shape{4});
  Tensor wrong_type({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
  Tensor scalar({1.0f}, Shape{});
  Tensor boolean({true, false}, Shape{2});

  EXPECT_THROW(matrix.add_nr(wrong_shape), std::invalid_argument);
  EXPECT_THROW(matrix.add_nr(wrong_type), std::invalid_argument);
  EXPECT_THROW(scalar += matrix, std::invalid_argument);
  EXPECT_THROW(boolean.add_nr(boolean), std::invalid_argument);
}

TEST(TensorUnaryNonRecording, NegateSquareAndReluSupportNumericDTypes) {
  Tensor floating({-2.0f, -0.0f, 3.0f}, Shape{3}, true);
  expect_values<float>(floating.negate_nr(), {2.0f, 0.0f, -3.0f});
  expect_values<float>(floating.square_nr(), {4.0f, 0.0f, 9.0f});
  expect_values<float>(floating.relu_nr(), {0.0f, 0.0f, 3.0f});
  EXPECT_FALSE(floating.relu_nr().requires_grad());

  Tensor integers({-2, 0, 3}, Shape{3});
  expect_values<int32_t>(integers.negate_nr(), {2, 0, -3});
  expect_values<int32_t>(integers.relu_nr(), {0, 0, 3});
}

TEST(TensorUnaryErrors, NegateAndReluRejectBool) {
  Tensor boolean({true, false}, Shape{2});
  EXPECT_THROW(boolean.negate_nr(), std::invalid_argument);
  EXPECT_THROW(boolean.relu_nr(), std::invalid_argument);
}

TEST(TensorReductions, ReduceScalarsVectorsMatricesAndHigherRanks) {
  Tensor scalar({7.0}, Shape{});
  expect_values<double>(scalar.sum_nr(), {7});
  expect_values<double>(scalar.mean(), {7});

  Tensor vector({-2, 5, 1, 4}, Shape{4});
  expect_values<int32_t>(vector.sum_nr(), {8});
  expect_values<int32_t>(vector.min(), {-2});
  expect_values<int32_t>(vector.max(), {5});

  Tensor rank4({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
               Shape{2, 1, 2, 2});
  expect_values_near<float>(rank4.sum_nr(), {36.0f}, 1e-6);
  expect_values_near<float>(rank4.mean(), {4.5f}, 1e-6);
  expect_values<float>(rank4.min(), {1.0f});
  expect_values<float>(rank4.max(), {8.0f});
}

TEST(TensorReductions, OperateOnNonContiguousLogicalValues) {
  Tensor transposed =
      Tensor({1.0, 5.0, -2.0, 4.0, 3.0, 6.0}, Shape{2, 3}).transpose_nr();
  expect_values<double>(transposed.sum_nr(), {17});
  expect_values<double>(transposed.mean(), {17.0 / 6.0});
  expect_values<double>(transposed.min(), {-2});
  expect_values<double>(transposed.max(), {6});
}

TEST(TensorReductionErrors, HandleEmptyAndUnsupportedDTypes) {
  Tensor empty = Tensor::zeros(Shape{0}, DType::F32);
  expect_values<float>(empty.sum_nr(), {0.0f});
  EXPECT_THROW(empty.mean(), std::invalid_argument);
  EXPECT_THROW(empty.min(), std::invalid_argument);
  EXPECT_THROW(empty.max(), std::invalid_argument);

  Tensor integers({1, 2}, Shape{2});
  EXPECT_THROW(integers.mean(), std::invalid_argument);
  Tensor boolean({true, false}, Shape{2});
  EXPECT_THROW(boolean.sum_nr(), std::invalid_argument);
  EXPECT_THROW(boolean.mean(), std::invalid_argument);
  EXPECT_THROW(boolean.min(), std::invalid_argument);
}

TEST(TensorMatMulNonRecording, MultipliesTwoDimensionalMatrices) {
  Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, true);
  Tensor b({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, Shape{3, 2});
  Tensor result = a.matmul_nr(b);

  expect_shape(result, {2, 2});
  expect_values_near<float>(result, {58, 64, 139, 154}, 1e-5);
  EXPECT_FALSE(result.requires_grad());
}

TEST(TensorMatMulNonRecording, SupportsBatchedAndBroadcastedTwoDimensionalOperand) {
  Tensor batched({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                 Shape{2, 2, 2});
  Tensor identity({1.0, 0.0, 0.0, 1.0}, Shape{2, 2});

  Tensor right_broadcast = batched.matmul_nr(identity);
  Tensor left_broadcast = identity.matmul_nr(batched);
  expect_shape(right_broadcast, {2, 2, 2});
  expect_values_near<double>(right_broadcast, {1, 2, 3, 4, 5, 6, 7, 8},
                             1e-10);
  expect_values_near<double>(left_broadcast, {1, 2, 3, 4, 5, 6, 7, 8},
                             1e-10);
}

TEST(TensorMatMulNonRecording, MaterializesNonContiguousOperands) {
  Tensor a =
      Tensor({1.0, 3.0, 2.0, 4.0}, Shape{2, 2}).transpose_nr();
  Tensor identity({1.0, 0.0, 0.0, 1.0}, Shape{2, 2});
  expect_values_near<double>(a.matmul_nr(identity), {1, 2, 3, 4}, 1e-10);
}

TEST(TensorMatMulErrors, RejectsUnsupportedShapesTypesAndDtypes) {
  Tensor vector({1.0f, 2.0f}, Shape{2});
  Tensor matrix({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
  EXPECT_THROW(vector.matmul_nr(matrix), std::invalid_argument);

  Tensor incompatible({1.0f, 2.0f, 3.0f}, Shape{3, 1});
  EXPECT_THROW(matrix.matmul_nr(incompatible), std::invalid_argument);

  Tensor doubles({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
  EXPECT_THROW(matrix.matmul_nr(doubles), std::invalid_argument);

  Tensor integers({1, 2, 3, 4}, Shape{2, 2});
  EXPECT_THROW(integers.matmul_nr(integers), std::invalid_argument);

  Tensor batch_a = Tensor::ones<float>(Shape{2, 2, 2});
  Tensor batch_b = Tensor::ones<float>(Shape{3, 2, 2});
  EXPECT_THROW(batch_a.matmul_nr(batch_b), std::invalid_argument);
}

TEST(TensorRecordingOps, AttachExpectedNodesOnlyWhenRequired) {
  Tensor leaf({1.0, 2.0}, Shape{2}, true);
  Tensor constant({3.0, 4.0}, Shape{2});

  Tensor add = leaf + constant;
  Tensor sub = leaf - constant;
  Tensor mul = leaf * constant;
  Tensor neg = -leaf;
  Tensor square = leaf.square();
  Tensor relu = leaf.relu();
  Tensor sum = leaf.sum();

  ASSERT_NE(add.grad_fn(), nullptr);
  EXPECT_EQ(add.grad_fn()->function(), "Addition_fn");
  ASSERT_NE(sub.grad_fn(), nullptr);
  EXPECT_EQ(sub.grad_fn()->function(), "Subtraction_fn");
  ASSERT_NE(mul.grad_fn(), nullptr);
  EXPECT_EQ(mul.grad_fn()->function(), "Multiplication_fn");
  EXPECT_NE(neg.grad_fn(), nullptr);
  EXPECT_NE(square.grad_fn(), nullptr);
  EXPECT_NE(relu.grad_fn(), nullptr);
  EXPECT_NE(sum.grad_fn(), nullptr);

  Tensor no_grad = constant + constant;
  EXPECT_EQ(no_grad.grad_fn(), nullptr);
  EXPECT_FALSE(no_grad.requires_grad());
}

TEST(TensorRecordingOps, RecordingAndNonRecordingForwardValuesMatch) {
  Tensor a({1.0, -2.0, 3.0, 4.0}, Shape{2, 2}, true);
  Tensor b({2.0, 3.0, 4.0, 5.0}, Shape{2, 2});

  expect_values<double>(a + b, {3, 1, 7, 9});
  expect_values<double>(a - b, {-1, -5, -1, -1});
  expect_values<double>(a * b, {2, -6, 12, 20});
  expect_values<double>(-a, {-1, 2, -3, -4});
  expect_values<double>(a.square(), {1, 4, 9, 16});
  expect_values<double>(a.relu(), {1, 0, 3, 4});
  expect_values<double>(a.sum(), {6});
}

TEST(TensorRecordingMatMulAndTranspose, AttachNodesAndPreserveValues) {
  Tensor a({1.0, 2.0, 3.0, 4.0}, Shape{2, 2}, true);
  Tensor identity({1.0, 0.0, 0.0, 1.0}, Shape{2, 2});
  Tensor product = a.matmul(identity);
  Tensor transposed = a.transpose(0, 1);

  EXPECT_NE(product.grad_fn(), nullptr);
  expect_values_near<double>(product, {1, 2, 3, 4}, 1e-10);
  EXPECT_NE(transposed.grad_fn(), nullptr);
  expect_values<double>(transposed, {1, 3, 2, 4});
}

} // namespace
} // namespace mango
