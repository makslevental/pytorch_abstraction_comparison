# LOL https://forums.developer.nvidia.com/t/cuda-10-1-nvidia-youre-now-fixing-gcc-bugs-that-gcc-doesnt-even-have/71063/11
#   usr/include/c++/8/bits/basic_string.tcc:1067
#   __p->_M_set_sharable(); -> (*__p)._M_set_sharable();

# you also have to add this at the top of the opencv CMake
project(cudnn_nets)

set(CUDNN_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu/libcudnn.so")
set(CUDA_cublas_LIBRARY "/usr/lib/x86_64-linux-gnu/libcublas.so")
set(CUDA_cublas_device_LIBRARY "/usr/lib/x86_64-linux-gnu/libcublas.so")
set(CMAKE_PREFIX_PATH "/home/maksim/dev_projects/cpp/cudnn")

find_package(CUDA REQUIRED)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_61,code=sm_61;-g;-lineinfo;-std=c++14;-Xcompiler;-ggdb)
else ()
    message("Release mode")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_61,code=sm_61;-O3;-DNDEBUG;-std=c++14;-Xcompiler;-DNDEBUG)
endif ()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
include_directories($ENV{CUDNN_PATH} $ENV{CUDNN_PATH}/include)
include_directories("include")
link_directories($ENV{CUDNN_PATH} $ENV{CUDNN_PATH}/lib $ENV{CUDNN_PATH}/lib64)

file(GLOB_RECURSE cudnn_SRC CONFIGURE_DEPENDS "src/*.cpp" "src/*.cu")
cuda_add_executable(main_cudnn ${cudnn_SRC})


#cuda_add_executable(main_cudnn
#        src/main.cu
#        src/cross_entropy_loss.cu
#        src/network.cpp
#
#        src/datasets/cifar10.cpp
#        src/datasets/dataset.cpp
#        src/datasets/mnist.cpp
#        src/datasets/stl10.cpp
#
#        src/layers/activation.cu
#        src/layers/addition.cu
#        src/layers/batch_norm_2d.cu
#        src/layers/conv_2d.cu
#        src/layers/dense.cu
#        src/layers/layer.cu
#        src/layers/pooling.cu
#        src/layers/softmax.cu
#        )

cuda_add_cublas_to_target(main_cudnn)

set_target_properties(main_cudnn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(main_cudnn PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
target_link_libraries(main_cudnn cudnn)
set_property(TARGET main_cudnn PROPERTY CXX_STANDARD 17)
