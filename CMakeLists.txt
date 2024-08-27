cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

project(tensorrt_bench LANGUAGES CXX CUDA)


find_package(CUDA  REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TensorRT_INCLUDE_DIRS})


find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSERS NAMES nvonnxparser)

link_directories(/usr/local/cuda/lib64)


# EXECUTABLE
add_executable(${PROJECT_NAME} tensorrt_bench.cpp ioHelper.cpp)
target_link_libraries(${PROJECT_NAME} nvinfer nvonnxparser cudart)
