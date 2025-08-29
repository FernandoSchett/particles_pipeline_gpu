cmake_minimum_required(VERSION 3.22)
project(P_SFC VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20) # Set CPP version
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # No compiler extensions
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75;80;86)
endif()

# Dependencies
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Boost REQUIRED)
find_package(CUDAToolkit REQUIRED)


add_subdirectory(third_party)
add_subdirectory(libs)
add_subdirectory(src)

# TO DO:
# Make role project download especifc versions of Boost and random, not as submodules. compile boost with the hole project. 