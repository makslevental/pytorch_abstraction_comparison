cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# help debugging Makefile cmd if ON
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Save libs and executables in the same place
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Output directory for applications")

include_directories("generic_utils")
add_subdirectory(cudnn)
add_subdirectory(lib_torch)