cmake_minimum_required(VERSION 3.18)
project(MiniGPT CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 75 80 90)

set(SOURCES main.cpp)
include_directories(${PROJECT_SOURCE_DIR})
add_executable(MiniGPT ${SOURCES})
set_target_properties(MiniGPT PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
