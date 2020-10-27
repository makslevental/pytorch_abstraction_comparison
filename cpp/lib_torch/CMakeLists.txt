# LOL https://forums.developer.nvidia.com/t/cuda-10-1-nvidia-youre-now-fixing-gcc-bugs-that-gcc-doesnt-even-have/71063/11
#   usr/include/c++/8/bits/basic_string.tcc:1067
#   __p->_M_set_sharable(); -> (*__p)._M_set_sharable();

cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
# you also have to add this at the top of the opencv CMake
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
project(lib_torch)

set(CMAKE_PREFIX_PATH "/home/maksim/dev_projects/cpp/lib_torch" "/home/maksim/dev_projects/libtorch")
set(OpenCV_DIR "/home/maksim/dev_projects/opencv/build")

# https://github.com/pytorch/pytorch/issues/13541#issuecomment-436342394
set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main_libtorch main.cpp dcgan.h resnet.h util.h)
target_link_libraries(main_libtorch "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET main_libtorch PROPERTY CXX_STANDARD 14)
