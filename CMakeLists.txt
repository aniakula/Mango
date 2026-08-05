cmake_minimum_required(VERSION 3.16)
project(mango LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MANGO_SOURCES
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
  src/auto_grad_node_impl/transpose.cpp
)

add_library(mango_lib ${MANGO_SOURCES})
target_include_directories(mango_lib PUBLIC include)

add_executable(mango src/main.cpp)
target_link_libraries(mango PRIVATE mango_lib)

# Contiguous float elementwise kernels via Apple vDSP (toggle manually).
# Defaults ON on Apple, OFF elsewhere. Force OFF if requested on non-Apple.
option(MANGO_ENABLE_VDSP
       "Use Apple vDSP for contiguous F32/F64 elementwise ops"
       ${APPLE})
if(MANGO_ENABLE_VDSP AND NOT APPLE)
  message(WARNING "MANGO_ENABLE_VDSP requires Apple Accelerate; disabling")
  set(MANGO_ENABLE_VDSP OFF CACHE BOOL
      "Use Apple vDSP for contiguous F32/F64 elementwise ops" FORCE)
endif()
if(MANGO_ENABLE_VDSP)
  target_compile_definitions(mango_lib PUBLIC MANGO_ENABLE_VDSP=1)
else()
  target_compile_definitions(mango_lib PUBLIC MANGO_ENABLE_VDSP=0)
endif()

if(APPLE)
  target_compile_definitions(mango_lib PRIVATE ACCELERATE_NEW_LAPACK)
  target_link_libraries(mango_lib PUBLIC "-framework Accelerate")
else()
  find_package(BLAS REQUIRED)
  target_link_libraries(mango_lib PUBLIC ${BLAS_LIBRARIES})
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(mango_lib PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(mango PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(mango_tests
    tests/tensor_core_test.cpp
    tests/tensor_ops_test.cpp
    tests/tensor_autograd_test.cpp
  )
  target_link_libraries(mango_tests PRIVATE mango_lib GTest::gtest_main)
  target_compile_options(mango_tests PRIVATE
    $<$<CXX_COMPILER_ID:Clang,GNU>:-Wall;-Wextra;-Wpedantic>
  )

  include(GoogleTest)
  gtest_discover_tests(mango_tests)
endif()
