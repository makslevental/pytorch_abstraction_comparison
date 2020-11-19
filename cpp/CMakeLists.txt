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


set(OpenCV_DIR "../../opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


set(CUDNN_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu/libcudnn.so")
set(CUDA_cublas_LIBRARY "/usr/lib/x86_64-linux-gnu/libcublas.so")
set(CUDA_cublas_device_LIBRARY "/usr/lib/x86_64-linux-gnu/libcublas.so")
set(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

find_package(CUDA REQUIRED)
if (APPLE)
    include_directories(/usr/local/cuda/include/)
    # for MacOS X or iOS, watchOS, tvOS (since 3.10.3)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("Debug mode")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_61,code=sm_61;-g;-lineinfo;-std=c++14;-Xcompiler;-ggdb)
    else ()
        message("Release mode")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_61,code=sm_61;-O3;-DNDEBUG;-std=c++14;-Xcompiler;-DNDEBUG)
    endif ()
    set(NVML_LIB "")
else ()
    include_directories(/usr/local/cuda-11.1/include/)
    include_directories(/usr/local/cuda-11.1/targets/x86_64-linux/include/)
    find_library(NVML_LIB nvidia-ml)
    message("${NVML_LIB}")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("Debug mode")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_61,code=sm_61;-g;-lineinfo;-std=c++17;-Xcompiler;-ggdb)
    else ()
        message("Release mode")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_61,code=sm_61;-O3;-DNDEBUG;-std=c++17;-Xcompiler;-DNDEBUG)
    endif ()
endif ()


set(CUDA_PROPAGATE_HOST_FLAGS OFF)
include_directories($ENV{CUDNN_PATH} $ENV{CUDNN_PATH}/include)
link_directories($ENV{CUDNN_PATH} $ENV{CUDNN_PATH}/lib $ENV{CUDNN_PATH}/lib64)

include_directories("generic_utils/include")
file(GLOB_RECURSE generic_utils_SRC CONFIGURE_DEPENDS "generic_utils/*.cpp")

add_subdirectory(cudnn)
add_subdirectory(lib_torch)