cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
project(ProjectName)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# help debugging Makefile cmd if ON
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Save libs and executables in the same place
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Output directory for applications")

include_directories("generic_utils")
add_subdirectory(cudnn)
#add_subdirectory(lib_torch)