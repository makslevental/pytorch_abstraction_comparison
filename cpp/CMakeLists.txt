cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
# THIS HAS TO COME BEFORE THE PROJECT LINE
set(CMAKE_C_COMPILER "gcc-10")
set(CMAKE_CXX_COMPILER "g++-10")
# THIS HAS TO COME BEFORE THE PROJECT LINE
project(pytorch_abstraction)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# help debugging Makefile cmd if ON
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Save libs and executables in the same place
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Output directory for applications")

set(CUDNN_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu/libcudnn.so")
set(CUDA_cublas_LIBRARY "/usr/lib/x86_64-linux-gnu/libcublas.so")
set(CUDA_cublas_device_LIBRARY "/usr/lib/x86_64-linux-gnu/libcublas.so")
set(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

find_package(CUDA REQUIRED)
if (APPLE)
    # for MacOS X or iOS, watchOS, tvOS (since 3.10.3)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("Debug mode")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_61,code=sm_61;-g;-lineinfo;-std=c++14;-Xcompiler;-ggdb)
    else ()
        message("Release mode")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_61,code=sm_61;-O3;-DNDEBUG;-std=c++14;-Xcompiler;-DNDEBUG)
    endif ()
else ()
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("Debug mode")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_61,code=sm_61;-g;-lineinfo;-std=c++17;-Xcompiler;-ggdb)
    else ()
        message("Release mode")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_61,code=sm_61;-O3;-DNDEBUG;-std=c++17;-Xcompiler;-DNDEBUG)
    endif ()
endif ()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
include_directories(/usr/local/cuda/include/)
include_directories($ENV{CUDNN_PATH} $ENV{CUDNN_PATH}/include)
link_directories($ENV{CUDNN_PATH} $ENV{CUDNN_PATH}/lib $ENV{CUDNN_PATH}/lib64)

include_directories("generic_utils")
add_subdirectory(cudnn)
add_subdirectory(lib_torch)