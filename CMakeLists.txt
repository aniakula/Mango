cmake_minimum_required(VERSION 3.16)
project(mango LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(mango
  src/main.cpp
  src/tensor_lib_impl/shape.cpp
  src/tensor_lib_impl/storage.cpp
  src/tensor_lib_impl/tensor.cpp
  src/tensor_lib_impl/tensor_internal.cpp
  src/auto_grad_node_impl/add.cpp
  src/auto_grad_node_impl/sub.cpp
  src/auto_grad_node_impl/mult.cpp
  src/auto_grad_node_impl/sq.cpp
  src/auto_grad_node_impl/sum.cpp
  src/auto_grad_node_impl/mean.cpp
  src/auto_grad_node_impl/matmul.cpp
  src/auto_grad_node_impl/relu.cpp
  src/auto_grad_node_impl/softmax.cpp
)

target_include_directories(mango PRIVATE include)

if(APPLE)
  target_compile_definitions(mango PRIVATE ACCELERATE_NEW_LAPACK)
  target_link_libraries(mango PRIVATE "-framework Accelerate")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(mango PRIVATE -Wall -Wextra -Wpedantic)
endif()
