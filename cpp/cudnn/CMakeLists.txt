# LOL https://forums.developer.nvidia.com/t/cuda-10-1-nvidia-youre-now-fixing-gcc-bugs-that-gcc-doesnt-even-have/71063/11
#   usr/include/c++/8/bits/basic_string.tcc:1067
#   __p->_M_set_sharable(); -> (*__p)._M_set_sharable();

# you also have to add this at the top of the opencv CMake
project(cudnn_nets)

include_directories("include")
file(GLOB_RECURSE cudnn_SRC CONFIGURE_DEPENDS "src/*.cpp" "src/*.cu")
cuda_add_executable(main_cudnn ${cudnn_SRC})

cuda_add_cublas_to_target(main_cudnn)

set_target_properties(main_cudnn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(main_cudnn PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
target_link_libraries(main_cudnn cudnn ${CMAKE_DL_LIBS})
set_property(TARGET main_cudnn PROPERTY CXX_STANDARD 17)
