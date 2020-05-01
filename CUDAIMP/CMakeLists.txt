cmake_minimum_required(VERSION 3.16)
project(CUDAIMP CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CUDAIMP main.cu types.h)

set_target_properties(
        CUDAIMP
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)