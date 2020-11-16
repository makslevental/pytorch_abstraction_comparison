# LOL https://forums.developer.nvidia.com/t/cuda-10-1-nvidia-youre-now-fixing-gcc-bugs-that-gcc-doesnt-even-have/71063/11
#   usr/include/c++/8/bits/basic_string.tcc:1067
#   __p->_M_set_sharable(); -> (*__p)._M_set_sharable();

# you also have to add this at the top of the opencv CMake
project(lib_torch)

set(Torch_DIR "../../../libtorch/share/cmake/Torch/")
set(OpenCV_DIR "../../../opencv/build")

# https://github.com/pytorch/pytorch/issues/13541#issuecomment-436342394
set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("include/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
file(GLOB_RECURSE libtorch_SRC CONFIGURE_DEPENDS src/*.cpp)
add_executable(main_libtorch ${libtorch_SRC})
target_link_libraries(main_libtorch "${TORCH_LIBRARIES}" "${OpenCV_LIBS}" ${CMAKE_DL_LIBS})
set_property(TARGET main_libtorch PROPERTY CXX_STANDARD 17)
