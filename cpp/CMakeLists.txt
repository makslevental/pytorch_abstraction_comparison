cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch_abstraction_comparison/)

set(CUDNN_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu/libcudnn.so")
set(CUDA_cublas_LIBRARY "/usr/lib/x86_64-linux-gnu/libcublas.so")
set(CMAKE_PREFIX_PATH "/home/maksim/dev_projects/libtorch")

# https://github.com/pytorch/pytorch/issues/13541#issuecomment-436342394
set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#message(FATAL_ERROR "${CUDNN_LIBRARY_PATH}")


add_executable(dcgan dcgan.cpp)
target_link_libraries(dcgan "${TORCH_LIBRARIES}")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 14)
