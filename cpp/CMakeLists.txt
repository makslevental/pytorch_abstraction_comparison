# LOL https://forums.developer.nvidia.com/t/cuda-10-1-nvidia-youre-now-fixing-gcc-bugs-that-gcc-doesnt-even-have/71063/11
#   usr/include/c++/8/bits/basic_string.tcc:1067
#   __p->_M_set_sharable(); -> (*__p)._M_set_sharable();

cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
# you also have to add this at the top of the opencv CMake
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
project(pytorch_abstraction_comparison)

set(CUDNN_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu/libcudnn.so")
set(CUDA_cublas_LIBRARY "/usr/lib/x86_64-linux-gnu/libcublas.so")
set(CUDA_cublas_device_LIBRARY "/usr/lib/x86_64-linux-gnu/libcublas.so")
set(CMAKE_PREFIX_PATH "/home/maksim/dev_projects/cpp" "/home/maksim/dev_projects/libtorch")
set(OpenCV_DIR "/home/maksim/dev_projects/opencv/build")

# https://github.com/pytorch/pytorch/issues/13541#issuecomment-436342394
set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_50,code=compute_50;-g;-lineinfo;-Xcompiler;-ggdb)
else()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_50,code=compute_50;-O3;-DNDEBUG;-Xcompiler;-DNDEBUG)
endif()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
include_directories($ENV{CUDNN_PATH} $ENV{CUDNN_PATH}/include)
link_directories($ENV{CUDNN_PATH} $ENV{CUDNN_PATH}/lib $ENV{CUDNN_PATH}/lib64)
cuda_add_executable(main resnet_cudnn.cu readubyte.cpp)
cuda_add_cublas_to_target(main)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#message(FATAL_ERROR "${CUDNN_LIBRARY_PATH}")

#add_executable(main main.cpp dcgan.h resnet.h util.h resnet_cudnn.h helper_cuda.h readubyte.h)
#target_link_libraries(main "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
target_link_libraries(main cudnn)
set_property(TARGET main PROPERTY CXX_STANDARD 14)
