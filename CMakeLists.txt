cmake_minimum_required(VERSION 3.16)
project(mango LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(mango
  src/main.cpp
  src/shape.cpp
  src/storage.cpp
  src/tensor.cpp
)

target_include_directories(mango PRIVATE include)

if(APPLE)
  target_compile_definitions(mango PRIVATE ACCELERATE_NEW_LAPACK)
  target_link_libraries(mango PRIVATE "-framework Accelerate")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(mango PRIVATE -Wall -Wextra -Wpedantic)
endif()
