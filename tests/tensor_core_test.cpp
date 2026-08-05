#include "tensor_test_utils.h"

#include "tensor_lib_headers/tensor_internal.h"

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>

namespace mango {
namespace {

using test::expect_shape;
using test::expect_values;
using test::expect_values_near;

TEST(TensorConstruction, InitializerPreservesShapeTypeAndValues) {
  Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});

  expect_shape(tensor, {2, 2});
  EXPECT_EQ(tensor.strides().to_vector(), (std::vector<size_t>{2, 1}));
  EXPECT_EQ(tensor.dtype(), DType::F32);
  EXPECT_EQ(tensor.numel(), 4u);
  EXPECT_EQ(tensor.storage_offset(), 0u);
  EXPECT_TRUE(tensor.is_contiguous());
  EXPECT_FALSE(tensor.requires_grad());
  EXPECT_EQ(tensor.grad(), nullptr);
  EXPECT_EQ(tensor.grad_fn(), nullptr);
  expect_values<float>(tensor, {1.0f, 2.0f, 3.0f, 4.0f});
}

TEST(TensorConstruction, SupportsAllDTypesAndScalarShape) {
  Tensor f64({2.5}, Shape{});
  Tensor i32({int32_t{-2}, int32_t{7}}, Shape{2});
  Tensor i64({int64_t{4}, int64_t{-9}}, Shape{2});
  Tensor boolean({true, false}, Shape{2});

  EXPECT_EQ(f64.dtype(), DType::F64);
  EXPECT_EQ(f64.shape().rank(), 0u);
  EXPECT_EQ(f64.numel(), 1u);
  EXPECT_EQ(i32.dtype(), DType::I32);
  EXPECT_EQ(i64.dtype(), DType::I64);
  EXPECT_EQ(boolean.dtype(), DType::B);
  expect_values<double>(f64, {2.5});
  expect_values<int32_t>(i32, {-2, 7});
  expect_values<int64_t>(i64, {4, -9});
  expect_values<bool>(boolean, {true, false});
}

TEST(TensorConstruction, RejectsInitializerSizeMismatch) {
  EXPECT_THROW((Tensor({1.0f, 2.0f}, Shape{3})), std::invalid_argument);
}

TEST(TensorFactories, ZerosAndOnesHonorShapeAndDType) {
  Tensor zeros = Tensor::zeros(Shape{2, 3}, DType::I64);
  Tensor ones = Tensor::ones<float>(Shape{2, 1, 2});

  expect_values<int64_t>(zeros, {0, 0, 0, 0, 0, 0});
  expect_values<float>(ones, {1, 1, 1, 1});
  expect_shape(ones, {2, 1, 2});
}

TEST(TensorFactories, EmptyShapeWithZeroDimensionHasNoElements) {
  Tensor empty = Tensor::zeros(Shape{2, 0, 3});
  EXPECT_EQ(empty.numel(), 0u);
  EXPECT_EQ(empty.data(), nullptr);
  expect_shape(empty, {2, 0, 3});
}

TEST(TensorFactories, RandnReturnsFiniteValuesAndRequestedMetadata) {
  Tensor tensor = Tensor::randn<double>(Shape{4, 5}, true);
  expect_shape(tensor, {4, 5});
  EXPECT_EQ(tensor.dtype(), DType::F64);
  EXPECT_TRUE(tensor.requires_grad());
  ASSERT_NE(tensor.grad(), nullptr);
  for (double value : test::values<double>(tensor)) {
    EXPECT_TRUE(std::isfinite(value));
  }
}

TEST(TensorCasting, ConvertsEveryElementAcrossDTypes) {
  Tensor source({-2.75, 0.0, 3.5}, Shape{3});

  Tensor f32 = source.to(DType::F32);
  Tensor i32 = source.to(DType::I32);
  Tensor i64 = source.to(DType::I64);
  Tensor boolean = source.to(DType::B);

  expect_values_near<float>(f32, {-2.75f, 0.0f, 3.5f}, 1e-6);
  expect_values<int32_t>(i32, {-2, 0, 3});
  expect_values<int64_t>(i64, {-2, 0, 3});
  expect_values<bool>(boolean, {true, false, true});
  EXPECT_EQ(source.dtype(), DType::F64);
}

TEST(TensorCasting, PreservesLogicalOrderOfNonContiguousTensor) {
  Tensor transposed =
      Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3}).transpose_nr();
  Tensor converted = transposed.to(DType::I32);
  expect_shape(converted, {3, 2});
  expect_values<int32_t>(converted, {1, 4, 2, 5, 3, 6});
}

TEST(TensorView, SharesStorageAndChangesOnlyMetadata) {
  Tensor tensor({1, 2, 3, 4, 5, 6}, Shape{2, 3});
  Tensor view = tensor.view(Shape{3, 2});

  expect_shape(view, {3, 2});
  EXPECT_EQ(view.strides().to_vector(), (std::vector<size_t>{2, 1}));
  static_cast<int *>(tensor.data())[2] = 99;
  expect_values<int>(view, {1, 2, 99, 4, 5, 6});
}

TEST(TensorView, RejectsElementCountChangeAndNonContiguousInput) {
  Tensor tensor({1, 2, 3, 4, 5, 6}, Shape{2, 3});
  EXPECT_THROW(tensor.view(Shape{4, 2}), std::invalid_argument);

  Tensor transposed = tensor.transpose_nr();
  EXPECT_THROW(transposed.view(Shape{6}), std::invalid_argument);
}

TEST(TensorReshape, ChangesMetadataInPlaceAndRejectsInvalidShapes) {
  Tensor tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
  tensor.reshape(Shape{4});
  expect_shape(tensor, {4});
  expect_values<double>(tensor, {1, 2, 3, 4});
  EXPECT_THROW(tensor.reshape(Shape{5}), std::invalid_argument);

  Tensor transposed = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2}).transpose_nr();
  EXPECT_THROW(transposed.reshape(Shape{4}), std::invalid_argument);
}

TEST(TensorClone, OwnsIndependentStorageAndClearsAutograd) {
  Tensor source({1.0, 2.0, 3.0, 4.0}, Shape{2, 2}, true);
  Tensor clone = source.clone();
  static_cast<double *>(clone.data())[0] = 100.0;

  expect_values<double>(source, {1, 2, 3, 4});
  expect_values<double>(clone, {100, 2, 3, 4});
  EXPECT_FALSE(clone.requires_grad());
  EXPECT_EQ(clone.grad(), nullptr);
}

TEST(TensorClone, CanCloneWithACompatibleShape) {
  Tensor source({1, 2, 3, 4}, Shape{2, 2});
  Tensor clone = source.clone(Shape{4, 1});
  expect_shape(clone, {4, 1});
  expect_values<int>(clone, {1, 2, 3, 4});
  EXPECT_THROW(source.clone(Shape{3}), std::invalid_argument);
}

TEST(TensorTranspose, HandlesArbitraryDimensionsAndContiguousMaterialization) {
  Tensor tensor({1, 2, 3, 4, 5, 6, 7, 8}, Shape{2, 2, 2});
  Tensor transposed = tensor.transpose_nr(0, 2);

  expect_shape(transposed, {2, 2, 2});
  EXPECT_EQ(transposed.strides().to_vector(), (std::vector<size_t>{1, 2, 4}));
  EXPECT_FALSE(transposed.is_contiguous());
  expect_values<int>(transposed, {1, 5, 3, 7, 2, 6, 4, 8});

  Tensor materialized = transposed.contiguous();
  EXPECT_TRUE(materialized.is_contiguous());
  expect_values<int>(materialized, {1, 5, 3, 7, 2, 6, 4, 8});
}

TEST(TensorTranspose, RejectsDimensionsOutsideRank) {
  Tensor tensor({1.0f, 2.0f}, Shape{2});
  EXPECT_THROW(tensor.transpose_inplace(0, 1), std::out_of_range);
  EXPECT_THROW(tensor.transpose_nr(2, 0), std::out_of_range);
}

TEST(TensorContiguous, PreservesLogicalValuesAndDoesNotAliasMaterializedCopy) {
  Tensor source({1, 2, 3, 4, 5, 6}, Shape{2, 3});
  Tensor transposed = source.transpose_nr();
  Tensor contiguous = transposed.contiguous();
  static_cast<int *>(contiguous.data())[0] = 50;

  expect_values<int>(transposed, {1, 4, 2, 5, 3, 6});
  expect_values<int>(contiguous, {50, 4, 2, 5, 3, 6});
}

TEST(TensorGradBuffer, AccumulatesZerosAndReducesBroadcastGradientForScalar) {
  Tensor vector({1.0, 2.0, 3.0}, Shape{3}, true);
  vector.accum_grad(Tensor({0.5, 1.0, 1.5}, Shape{3}));
  vector.accum_grad(Tensor({0.5, 1.0, 1.5}, Shape{3}));
  test::expect_grad_near<double>(vector, {1.0, 2.0, 3.0});
  vector.zero_grad();
  test::expect_grad_near<double>(vector, {0.0, 0.0, 0.0});

  Tensor scalar({2.0}, Shape{}, true);
  scalar.accum_grad(Tensor({1.0, 2.0, 3.0}, Shape{3}));
  test::expect_grad_near<double>(scalar, {6.0});
}

TEST(TensorLogging, IncludesMetadataAndLogicalValues) {
  Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
  std::ostringstream stream;
  tensor.log(stream);
  const std::string output = stream.str();
  EXPECT_NE(output.find("dtype=f32"), std::string::npos);
  EXPECT_NE(output.find("shape=(2, 2)"), std::string::npos);
  EXPECT_NE(output.find("contiguous=true"), std::string::npos);
  EXPECT_NE(output.find("4"), std::string::npos);
}

TEST(TensorInternalAssertions, RejectShapeAndTypeMismatches) {
  EXPECT_NO_THROW(detail::assert_same_shape(Shape{2, 3}, Shape{2, 3}));
  EXPECT_THROW(detail::assert_same_shape(Shape{2, 3}, Shape{6}),
               std::invalid_argument);
  EXPECT_THROW(detail::assert_same_shape(Shape{2, 3}, Shape{3, 2}),
               std::invalid_argument);
  EXPECT_THROW(detail::assert_same_type(DType::F32, DType::F64),
               std::invalid_argument);
}

} // namespace
} // namespace mango
